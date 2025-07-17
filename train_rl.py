import os
import torch
import numpy as np
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Any
import wandb
from datetime import datetime

from ppo_trainer import PPOTrainer, create_structured_prompt

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_batch(dataset, indices, char_limits: Dict[str, int]) -> tuple:
    """Prepare a batch of examples from the dataset"""
    prompts = []
    reference_codes = []
    
    for idx in indices:
        example = dataset[idx]
        prompt = create_structured_prompt(
            example, 
            char_limit=char_limits['context'], 
            ccl_char_limit=char_limits['ccl']
        )
        ref_code = example['src_code'][:char_limits['generated']]
        
        prompts.append(prompt)
        reference_codes.append(ref_code)
    
    return prompts, reference_codes

def train(args):
    """Main training function"""
    # Initialize wandb if enabled
    if args.use_wandb:
        run_name = f"ppo-{args.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        wandb.init(project="et-code-rl", name=run_name, config=vars(args))
    
    # Set random seed
    set_seed(args.seed)
    
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset['train']
    val_dataset = dataset['validation'] if 'validation' in dataset else dataset['train'].select(range(100, 120))
    
    # Character limits for context and generation
    char_limits = {
        'context': args.context_char_limit,
        'ccl': args.ccl_char_limit,
        'generated': args.generated_char_limit
    }
    
    # Initialize trainer
    trainer = PPOTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        bleu_weight=args.bleu_weight,
        llm_weight=args.llm_weight,
        use_llm_judge=args.use_llm_judge,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True
        }
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        if load_checkpoint(trainer, args.resume_from):
            # Try to get epoch and step from checkpoint
            checkpoint = torch.load(args.resume_from, map_location=trainer.device)
            
            # Get epoch from checkpoint metadata if available
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch} (from checkpoint metadata)")
            # Otherwise try to extract from filename
            else:
                try:
                    if "epoch" in args.resume_from:
                        epoch_str = args.resume_from.split("epoch")[-1].split(".")[0]
                        start_epoch = int(epoch_str) + 1
                        print(f"Resuming from epoch {start_epoch} (from filename)")
                except:
                    print("Could not determine epoch from checkpoint filename. Starting from epoch 0.")
            
            # Get global step from checkpoint metadata if available
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                print(f"Resuming from global step {global_step}")
        else:
            print("Failed to resume from checkpoint. Starting from scratch.")
    else:
        # Save initial model checkpoint if not resuming
        initial_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-e0-b0.pt")
        save_checkpoint(trainer, initial_checkpoint_path, epoch=0, global_step=0)
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs")
    
    # Calculate number of batches
    dataset_size = len(train_dataset)
    num_batches = dataset_size // args.batch_size
    
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        print(f"Epoch {epoch+1}/{start_epoch + args.num_epochs}")
        
        # Shuffle dataset indices
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)
        
        epoch_rewards = []
        epoch_losses = []
        
        # Process batches
        for batch_idx in tqdm(range(num_batches)):
            # Get batch indices
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, dataset_size)
            batch_indices = all_indices[start_idx:end_idx]
            
            # Prepare batch
            prompts, reference_codes = prepare_batch(train_dataset, batch_indices, char_limits)
            
            # Train on batch
            metrics = trainer.train_step(prompts, reference_codes)
            
            # Log metrics
            epoch_rewards.append(metrics['mean_reward'])
            epoch_losses.append(metrics['total_loss'])
            
            # Calculate global step
            current_step = epoch * num_batches + batch_idx + global_step
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    "batch_reward": metrics['mean_reward'],
                    "batch_policy_loss": metrics['policy_loss'],
                    "batch_value_loss": metrics['value_loss'],
                    "batch_total_loss": metrics['total_loss'],
                    "batch": current_step
                })
            
            # Save checkpoint periodically
            if batch_idx % args.save_every == 0 and batch_idx > 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-e{epoch}-b{current_step}.pt")
                save_checkpoint(trainer, checkpoint_path, epoch=epoch, global_step=current_step)
        
        # Evaluate on validation set
        val_metrics = evaluate(trainer, val_dataset, args.eval_batch_size, char_limits)
        
        # Log epoch metrics
        mean_reward = np.mean(epoch_rewards)
        mean_loss = np.mean(epoch_losses)
        
        print(f"Epoch {epoch+1} stats:")
        print(f"  Train reward: {mean_reward:.4f}")
        print(f"  Train loss: {mean_loss:.4f}")
        print(f"  Val reward: {val_metrics['mean_reward']:.4f}")
        print(f"  Val BLEU: {val_metrics['mean_bleu']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_reward": mean_reward,
                "train_loss": mean_loss,
                "val_reward": val_metrics['mean_reward'],
                "val_bleu": val_metrics['mean_bleu'],
                "val_llm_score": val_metrics['mean_llm_score']
            })
        
        # Save epoch checkpoint
        current_step = (epoch + 1) * num_batches + global_step
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}.pt")
        save_checkpoint(trainer, checkpoint_path, epoch=epoch, global_step=current_step)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.model.save_pretrained(final_path)
    trainer.tokenizer.save_pretrained(final_path)
    
    print("Training complete!")

def evaluate(trainer, dataset, batch_size, char_limits):
    """Evaluate the model on a dataset"""
    all_rewards = []
    all_bleu_scores = []
    all_llm_scores = []
    
    # Sample random indices for evaluation
    eval_size = min(len(dataset), 20)  # Evaluate on 20 examples max
    eval_indices = random.sample(range(len(dataset)), eval_size)
    
    # Process in small batches
    for i in range(0, len(eval_indices), batch_size):
        batch_indices = eval_indices[i:i+batch_size]
        prompts, reference_codes = prepare_batch(dataset, batch_indices, char_limits)
        
        batch_rewards = []
        batch_bleu = []
        batch_llm = []
        
        # Generate responses and compute rewards
        for prompt, ref_code in zip(prompts, reference_codes):
            generated_text, _ = trainer.generate_response(prompt)
            
            # Get reward components
            reward_result = trainer.reward_system.get_reward(generated_text, ref_code)
            
            batch_rewards.append(reward_result['combined_reward'])
            batch_bleu.append(reward_result['bleu_score'])
            
            if 'llm_judge_score' in reward_result:
                batch_llm.append(reward_result['llm_judge_score'])
        
        all_rewards.extend(batch_rewards)
        all_bleu_scores.extend(batch_bleu)
        all_llm_scores.extend(batch_llm)
    
    # Compute mean metrics
    results = {
        'mean_reward': np.mean(all_rewards),
        'mean_bleu': np.mean(all_bleu_scores),
        'mean_llm_score': np.mean(all_llm_scores) if all_llm_scores else 0.0
    }
    
    return results

def save_checkpoint(trainer, path, epoch=None, global_step=None):
    """Save a checkpoint of the model and optimizer"""
    checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'value_net_state_dict': trainer.value_net.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict()
    }
    
    # Save training metadata if provided
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if global_step is not None:
        checkpoint['global_step'] = global_step
        
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(trainer, path):
    """Load a checkpoint into the model and optimizer"""
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found. Starting from scratch.")
        return False
    
    try:
        checkpoint = torch.load(path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Log metadata if available
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"Checkpoint from global step: {checkpoint['global_step']}")
            
        print(f"Checkpoint loaded from {path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a code generation model with PPO")
    
    # Model and dataset
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", 
                        help="Name of the pretrained model")
    parser.add_argument("--dataset_name", type=str, default="xinshuo/ET_code_with_context", 
                        help="Name of the dataset")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=3, 
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=2, 
                        help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-7, 
                        help="Learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=4, 
                        help="Number of PPO epochs per batch")
    
    # PPO hyperparameters
    parser.add_argument("--clip_epsilon", type=float, default=0.02, 
                        help="PPO clipping parameter")
    parser.add_argument("--value_coef", type=float, default=0.5, 
                        help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, 
                        help="Maximum gradient norm")
    
    # Reward settings
    parser.add_argument("--bleu_weight", type=float, default=0.8, 
                        help="Weight for BLEU score in reward")
    parser.add_argument("--llm_weight", type=float, default=0.2, 
                        help="Weight for LLM judge in reward")
    parser.add_argument("--use_llm_judge", action="store_true", 
                        help="Whether to use LLM judge")
    
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=400, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, 
                        help="Temperature for generation")
    
    # Character limits
    parser.add_argument("--context_char_limit", type=int, default=1000, 
                        help="Character limit for context")
    parser.add_argument("--ccl_char_limit", type=int, default=100, 
                        help="Character limit for CCL files")
    parser.add_argument("--generated_char_limit", type=int, default=500, 
                        help="Character limit for generated code")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=10, 
                        help="Save checkpoint every N batches")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use wandb for logging")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args) 