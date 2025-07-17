import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from combined_reward import CombinedReward
import numpy as np
from typing import List, Dict, Tuple, Any

class PPOTrainer:
    """
    PPO Trainer for fine-tuning language models with reinforcement learning
    using BLEU score and LLM judge as rewards.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        learning_rate: float = 1e-7,
        clip_epsilon: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 20,
        bleu_weight: float = 0.7,
        llm_weight: float = 0.3,
        use_llm_judge: bool = True,
        max_new_tokens: int = 400,
        temperature: float = 0.3,
        generation_config: Dict[str, Any] = None
    ):
        """
        Initialize the PPO trainer with model and hyperparameters.
        
        Args:
            model_name: Name of the HuggingFace model to use
            learning_rate: Learning rate for optimizer
            clip_epsilon: PPO clipping parameter
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per batch
            bleu_weight: Weight for BLEU score in reward
            llm_weight: Weight for LLM judge score in reward
            use_llm_judge: Whether to use LLM judge for reward
            generation_config: Configuration for text generation
        """
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        
        # Default generation config
        self.generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True
        }
        
        # Update with user config if provided
        if generation_config:
            self.generation_config.update(generation_config)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        # Initialize value network
        hidden_size = self.model.config.hidden_size
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(self.device)
        
        # Optimizer for both policy and value networks
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate
        )
        
        # Reward system
        self.reward_system = CombinedReward(
            bleu_weight=bleu_weight, 
            llm_weight=llm_weight, 
            use_llm_judge=use_llm_judge
        )
        
        # Set pad token ID if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def generate_response(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """
        Generate a response for a given prompt and return both the text and token sequence.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Tuple of (generated_text, full_token_sequence)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract generated tokens and decode
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up generated text
        if "```" in generated_text:
            generated_text = generated_text.split("```")[0].strip()
        if "\n" in generated_text and generated_text.find("\n") < 50:
            generated_text = generated_text[generated_text.find("\n")+1:]
        
        return generated_text, outputs[0]
    
    def get_log_probs_and_values(
        self, 
        sequences: List[torch.Tensor], 
        prompt_lengths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and value estimates for sequences.
        
        Args:
            sequences: List of token sequences
            prompt_lengths: List of prompt lengths for each sequence
            
        Returns:
            Tuple of (log_probs, values) tensors
        """
        log_probs = []
        values = []
        
        for seq, prompt_len in zip(sequences, prompt_lengths):
            # Add batch dimension
            seq = seq.unsqueeze(0)
            
            # Forward pass through model
            outputs = self.model(input_ids=seq, output_hidden_states=True)
            logits = outputs.logits[0]  # Remove batch dimension
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            
            # Get log probs for generated tokens only
            generated_logits = logits[prompt_len-1:-1]  # Shift for next token prediction
            generated_tokens = seq[0][prompt_len:]
            
            # Handle empty sequences gracefully
            if len(generated_tokens) > 0 and len(generated_logits) > 0:
                log_probs_seq = F.log_softmax(generated_logits, dim=-1)
                token_log_probs = log_probs_seq.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)
                log_probs.append(token_log_probs.mean())
            else:
                # Use a small log prob from the last logit to maintain gradient flow
                last_logit = logits[-1:, :]
                dummy_log_prob = F.log_softmax(last_logit, dim=-1)[0, 0]
                log_probs.append(dummy_log_prob)
            
            # Get value from last hidden state
            value = self.value_net(hidden_states[0, -1:])  # Last token
            values.append(value.squeeze())
            
            # Clean up to prevent memory accumulation
            if torch.is_grad_enabled():
                torch.cuda.empty_cache()
        
        return torch.stack(log_probs), torch.stack(values)
    
    def train_step(
        self, 
        prompts: List[str], 
        reference_codes: List[str]
    ) -> Dict[str, float]:
        """
        Perform a single PPO training step on a batch of examples.
        
        Args:
            prompts: List of input prompts
            reference_codes: List of reference code strings
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = len(prompts)
        
        # Generate responses and collect data
        sequences = []
        prompt_lengths = []
        rewards = []
        
        # Rollout phase: generate responses and compute rewards
        for prompt, ref_code in zip(prompts, reference_codes):
            # Generate response
            generated_text, full_sequence = self.generate_response(prompt)
            
            # Compute reward
            reward_result = self.reward_system.get_reward(generated_text, ref_code)
            reward = reward_result['combined_reward']
            
            # Store data
            sequences.append(full_sequence)
            prompt_lengths.append(len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]))
            rewards.append(reward)
        
        # Get old log probs and values (no gradients needed for baseline)
        with torch.no_grad():
            old_log_probs, old_values = self.get_log_probs_and_values(sequences, prompt_lengths)
        
        # Compute advantages and returns
        rewards_tensor = torch.tensor(rewards, device=self.device)
        advantages = rewards_tensor - old_values
        
        # Normalize advantages if batch size > 1
        if batch_size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = rewards_tensor.clone()
        
        # PPO optimization phase
        for epoch in range(self.ppo_epochs):
            # Get new log probs and values (with gradients)
            new_log_probs, new_values = self.get_log_probs_and_values(sequences, prompt_lengths)
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            # Compute PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Combined loss
            total_loss = policy_loss + self.value_coef * value_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            # check the change of parameters, print a few examples
            print(f"PPO Epoch {epoch}, Loss: {total_loss.item()}")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
        
        # Return metrics
        return {
            'mean_reward': torch.mean(rewards_tensor).item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

def create_structured_prompt(example, char_limit=10000, ccl_char_limit=1000):
    """Create prompt (same as in load_infer.py)"""
    context = example['context'][:char_limit] if len(example['context']) > char_limit else example['context']
    
    prompt = f"""<|system|>
You are an expert C/C++ developer working on EinsteinToolkit, a codebase for numerical relativity simulations.

<|user|>
Create C/C++ code for the file `{example['src_filename']}` in thorn `{example['thorn_name']}`.

## Thorn Information:
- Name: {example['thorn_name']}
- Target file: {example['src_filename']}

## Interface Definition in interface.ccl:
```
{example['interface'][:ccl_char_limit] + "...more..." if len(example['interface']) > ccl_char_limit else example['interface']}
```

## Schedule Definition in schedule.ccl:
```
{example['schedule'][:ccl_char_limit] + "...more..." if len(example['schedule']) > ccl_char_limit else example['schedule']}
```

## Parameters Definition in param.ccl:
```
{example['param'][:ccl_char_limit] + "...more..." if len(example['param']) > ccl_char_limit else example['param']}
```

## Configuration Definition in configuration.ccl:
```
{example['configuration'][:ccl_char_limit] + "...more..." if len(example['configuration']) > ccl_char_limit else example['configuration']}
```

## Related Code Context:
```
{context}
```

## Instructions:
Generate only the complete C/C++ source code for `{example['src_filename']}`. Include necessary headers, functions, and follow EinsteinToolkit conventions.

<|assistant|>
Here is the code for `{example['src_filename']}`:
```"""
    return prompt

def test_ppo():
    """Test PPO training with a small batch"""

    # we are constrained in computing resources, set upper limits
    context_char_limit = 1000
    ccl_char_limit = 100
    generated_char_limit = 500
    generated_max_tokens = 400


    trainer = PPOTrainer(
        bleu_weight=0.8, 
        llm_weight=0.2, 
        use_llm_judge=True,  # Disable for faster testing
        generation_config={
            "max_new_tokens": generated_max_tokens,
            "temperature": 0.3
        }
        # max_new_tokens=generated_max_tokens,
        # temperature=0.3
    )
    
    # Load dataset
    ds = load_dataset("xinshuo/ET_code_with_context")
    train_ds = ds['train']
    
    # Prepare small batch
    batch_indices = [1, 3, 5]
    prompts = []
    reference_codes = []
    
    for i in batch_indices:
        example = train_ds[i]
        prompt = create_structured_prompt(example, char_limit=context_char_limit, ccl_char_limit=ccl_char_limit)
        ref_code = example['src_code'][:generated_char_limit]
        
        prompts.append(prompt)
        reference_codes.append(ref_code)
    
    print("Running PPO training step...")
    metrics = trainer.train_step(prompts, reference_codes)
    
    print(f"Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    test_ppo() 