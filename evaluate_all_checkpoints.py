import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import torch
import gc
import wandb
import psutil

# Add parent directory to path to import from evaluate_GReX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_GReX import evaluate_grex

def find_checkpoints(checkpoint_dir, pattern="checkpoint-e*-b*.pt"):
    """Find all checkpoints matching the pattern in the directory"""
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    # Extract batch numbers for sorting
    checkpoint_info = []
    for path in checkpoint_paths:
        try:
            # Extract batch number from filename
            filename = os.path.basename(path)
            if "b" in filename:
                batch_str = filename.split("b")[1].split(".")[0]
                batch_num = int(batch_str)
                checkpoint_info.append((batch_num, path))
        except Exception as e:
            print(f"Error parsing checkpoint filename {path}: {e}")
    
    # Sort by batch number
    checkpoint_info.sort(key=lambda x: x[0])
    return checkpoint_info

def log_system_metrics():
    """Log system metrics including GPU and CPU utilization"""
    metrics = {}
    
    # CPU metrics
    metrics["cpu_percent"] = psutil.cpu_percent()
    metrics["memory_percent"] = psutil.virtual_memory().percent
    
    # GPU metrics if available
    if torch.cuda.is_available():
        metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        
        # Try to get GPU utilization if nvidia-smi is available
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
            gpu_utilization = float(result.decode('utf-8').strip())
            metrics["gpu_utilization_percent"] = gpu_utilization
        except:
            pass
    
    # Log to wandb
    wandb.log(metrics)
    
    return metrics

def evaluate_checkpoints(checkpoint_info, start_line, end_line, results_file=None):
    """Evaluate all checkpoints and collect results"""
    results = []
    
    for batch_num, checkpoint_path in tqdm(checkpoint_info, desc="Evaluating checkpoints"):
        print(f"\nEvaluating checkpoint from batch {batch_num}: {checkpoint_path}")
        
        # Log start of evaluation
        wandb.log({"batch": batch_num, "status": "evaluation_started"})
        
        # Log system metrics before evaluation
        pre_eval_metrics = log_system_metrics()
        print(f"Pre-evaluation metrics: {pre_eval_metrics}")
        
        # Evaluate the checkpoint
        eval_result = evaluate_grex(checkpoint_path, start_line, end_line)
        
        # Store results
        result_entry = {
            "batch": batch_num,
            "checkpoint_path": checkpoint_path,
            "compilation_success": eval_result["compilation_success"],
            "simulation_success": eval_result["simulation_success"],
            "plt_files_count": eval_result["plt_files_count"],
            "header_bleu_score": eval_result["header_bleu_score"],
            "rho_energy_similarity": eval_result["rho_energy_similarity"],
            "total_score": eval_result["total_score"]
        }
        
        results.append(result_entry)
        
        # Log evaluation results to wandb
        wandb.log({
            "batch": batch_num,
            "compilation_success": int(eval_result["compilation_success"]),
            "simulation_success": int(eval_result["simulation_success"]),
            "plt_files_count": eval_result["plt_files_count"],
            "header_bleu_score": eval_result["header_bleu_score"],
            "rho_energy_similarity": eval_result["rho_energy_similarity"],
            "total_score": eval_result["total_score"],
            "status": "evaluation_completed"
        })
        
        # Save results after each evaluation to handle potential crashes
        if results_file:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Explicitly clean up GPU memory after each evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log post-cleanup metrics
            post_cleanup_metrics = log_system_metrics()
            print(f"GPU memory after cleanup: {post_cleanup_metrics.get('gpu_memory_allocated_mb', 0):.2f} MB allocated, "
                  f"{post_cleanup_metrics.get('gpu_memory_reserved_mb', 0):.2f} MB reserved")
    
    return results

def plot_results(results, output_dir):
    """Generate plots from evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    batches = [r["batch"] for r in results]
    compilation_success = [int(r["compilation_success"]) for r in results]
    simulation_success = [int(r["simulation_success"]) for r in results]
    header_bleu_scores = [r["header_bleu_score"] for r in results]
    rho_energy_similarities = [r.get("rho_energy_similarity", 0.0) for r in results]  # Use get with default for backward compatibility
    total_scores = [r["total_score"] for r in results]
    
    # Plot compilation success
    plt.figure(figsize=(10, 6))
    plt.plot(batches, compilation_success, 'bo-', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('Compilation Success (0/1)')
    plt.title('Compilation Success vs. Training Progress')
    plt.grid(True)
    plt.yticks([0, 1])
    plt_path = os.path.join(output_dir, 'compilation_success.png')
    plt.savefig(plt_path)
    plt.close()
    wandb.log({"compilation_success_plot": wandb.Image(plt_path)})
    
    # Plot simulation success
    plt.figure(figsize=(10, 6))
    plt.plot(batches, simulation_success, 'go-', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('Simulation Success (0/1)')
    plt.title('Simulation Success vs. Training Progress')
    plt.grid(True)
    plt.yticks([0, 1])
    plt_path = os.path.join(output_dir, 'simulation_success.png')
    plt.savefig(plt_path)
    plt.close()
    wandb.log({"simulation_success_plot": wandb.Image(plt_path)})
    
    # Plot header BLEU score
    plt.figure(figsize=(10, 6))
    plt.plot(batches, header_bleu_scores, 'mo-', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('Header BLEU Score (0-1)')
    plt.title('Header BLEU Score vs. Training Progress')
    plt.grid(True)
    plt.ylim(0, 1)
    plt_path = os.path.join(output_dir, 'header_bleu_score.png')
    plt.savefig(plt_path)
    plt.close()
    wandb.log({"header_bleu_score_plot": wandb.Image(plt_path)})
    
    # Plot RHO_ENERGY similarity
    plt.figure(figsize=(10, 6))
    plt.plot(batches, rho_energy_similarities, 'co-', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('RHO_ENERGY Similarity (0-1)')
    plt.title('RHO_ENERGY Similarity vs. Training Progress')
    plt.grid(True)
    plt.ylim(0, 1)
    plt_path = os.path.join(output_dir, 'rho_energy_similarity.png')
    plt.savefig(plt_path)
    plt.close()
    wandb.log({"rho_energy_similarity_plot": wandb.Image(plt_path)})
    
    # Plot total score
    plt.figure(figsize=(10, 6))
    plt.plot(batches, total_scores, 'ro-', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('Total Score (0-1)')
    plt.title('Total Evaluation Score vs. Training Progress')
    plt.grid(True)
    plt.ylim(0, 1)
    plt_path = os.path.join(output_dir, 'total_score.png')
    plt.savefig(plt_path)
    plt.close()
    wandb.log({"total_score_plot": wandb.Image(plt_path)})
    
    # Combined plot
    plt.figure(figsize=(12, 8))
    plt.plot(batches, compilation_success, 'bo-', linewidth=2, label='Compilation Success')
    plt.plot(batches, simulation_success, 'go-', linewidth=2, label='Simulation Success')
    plt.plot(batches, header_bleu_scores, 'mo-', linewidth=2, label='Header BLEU Score')
    plt.plot(batches, rho_energy_similarities, 'co-', linewidth=2, label='RHO_ENERGY Similarity')
    plt.plot(batches, total_scores, 'ro-', linewidth=2, label='Total Score')
    plt.xlabel('Batch Number')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics vs. Training Progress')
    plt.grid(True)
    plt.legend()
    plt_path = os.path.join(output_dir, 'combined_metrics.png')
    plt.savefig(plt_path)
    plt.close()
    wandb.log({"combined_metrics_plot": wandb.Image(plt_path)})
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints and generate plots")
    parser.add_argument("--checkpoint_dir", type=str, default="/pscratch/sd/x/xinshuo/takehome/checkpoints",
                        help="Directory containing checkpoint files")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save results and plots")
    parser.add_argument("--start_line", type=int, required=True,
                        help="Starting line number of the section to replace")
    parser.add_argument("--end_line", type=int, required=True,
                        help="Ending line number of the section to replace")
    parser.add_argument("--batch_range", type=str, default="0,200,10",
                        help="Range of batches to evaluate in format 'start,end,step'")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file if available")
    parser.add_argument("--wandb_project", type=str, default="grex-evaluation",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (team) name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Results file path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    
    # Initialize wandb
    run_name = args.wandb_run_name or f"eval-{timestamp}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "checkpoint_dir": args.checkpoint_dir,
            "start_line": args.start_line,
            "end_line": args.end_line,
            "batch_range": args.batch_range,
        }
    )
    
    # Parse batch range
    batch_range = [int(x) for x in args.batch_range.split(",")]
    if len(batch_range) == 3:
        start, end, step = batch_range
    else:
        start, end = batch_range
        step = 10  # Default step
    
    # Find all checkpoints
    all_checkpoints = find_checkpoints(args.checkpoint_dir)
    
    # Filter checkpoints by batch range
    selected_checkpoints = [(batch, path) for batch, path in all_checkpoints 
                           if start <= batch <= end and (batch - start) % step == 0]
    
    if not selected_checkpoints:
        print(f"No checkpoints found in range {start}-{end} with step {step}")
        wandb.log({"status": "no_checkpoints_found"})
        wandb.finish()
        return
    
    print(f"Found {len(selected_checkpoints)} checkpoints to evaluate")
    for batch, path in selected_checkpoints:
        print(f"  Batch {batch}: {os.path.basename(path)}")
    
    # Check if we should resume from existing results
    if args.resume and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        print(f"Resuming from existing results with {len(existing_results)} entries")
        
        # Filter out checkpoints that have already been evaluated
        evaluated_batches = {r["batch"] for r in existing_results}
        selected_checkpoints = [(batch, path) for batch, path in selected_checkpoints 
                               if batch not in evaluated_batches]
        
        print(f"Remaining checkpoints to evaluate: {len(selected_checkpoints)}")
    else:
        existing_results = []
    
    # Log initial system metrics
    initial_metrics = log_system_metrics()
    print(f"Initial system metrics: {initial_metrics}")
    
    # Evaluate checkpoints
    if selected_checkpoints:
        new_results = evaluate_checkpoints(selected_checkpoints, args.start_line, args.end_line, results_file)
        all_results = existing_results + new_results
    else:
        all_results = existing_results
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    plot_results(all_results, args.output_dir)
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Log final system metrics
        final_metrics = log_system_metrics()
        print(f"Final GPU memory: {final_metrics.get('gpu_memory_allocated_mb', 0):.2f} MB allocated, "
              f"{final_metrics.get('gpu_memory_reserved_mb', 0):.2f} MB reserved")
    
    # Upload results file to wandb
    wandb.save(results_file)
    
    # Finish wandb run
    wandb.log({"status": "evaluation_complete"})
    wandb.finish()
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 