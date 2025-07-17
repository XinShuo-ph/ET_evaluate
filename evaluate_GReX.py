# using the function implemented in generate_GReX.py, generate and replace id.cpp file in /pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval/id.cpp

# go to /pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval
# make -j128 > log.txt 2>&1
# and check correct compilation (last word should be "SUCCESS", maybe we can use this creterion: looks for the last two lines, check if "SUCCESS" is contained), if so, we should give a reward for compilation



# if compilation passes, then copy the executable
# from /pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval/main2d.gnu.TPROF.MPI.OMP.CUDA.ex
# to /pscratch/sd/x/xinshuo/runGReX/20250707_eval/main2d.gnu.TPROF.MPI.OMP.CUDA.ex
# (remove the old one in /pscratch/sd/x/xinshuo/runGReX/20250707_eval/ if it exists)


# then we go to /pscratch/sd/x/xinshuo/runGReX/20250707_eval
# and run these commands:

# export CRAY_ACCEL_TARGET=nvidia80
# export MPICH_GPU_SUPPORT_ENABLED=1
# export CRAY_ACCEL_TARGET=nvidia80
# export AMREX_CUDA_ARCH=8.0
# export MPICH_GPU_SUPPORT_ENABLED=0
# export SLURM_CPU_BIND="cores"
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread
# srun -n 128 --ntasks-per-node=32  --gpu-bind=none ./main2d.gnu.TPROF.MPI.OMP.CUDA.ex inputs  > log.txt 2>&1


# after the run finishes, check if there are (and how many) plt0000[1-5] files in /pscratch/sd/x/xinshuo/runGReX/20250707_eval/
# if so, we should give a reward for running the code


# Then we check the plt files with the ground truth in /pscratch/sd/x/xinshuo/runGReX/20250707_eval/gt/plt0000[1-5]
# first, the header /pscratch/sd/x/xinshuo/runGReX/20250707_eval/plt00001/Header should be the same as /pscratch/sd/x/xinshuo/runGReX/20250707_eval/gt/plt00001/Header
# let's reward by BLEU score between the two headers


# we can also check specific variables. That would be a more ambitious task. We will do this later.

import os
import sys
import shutil
import glob
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
import torch
import gc
import numpy as np

# Try to import yt for data analysis
try:
    import yt
except ImportError:
    print("Warning: yt package not found. RHO_ENERGY comparison will be skipped.")
    yt = None

# Add parent directory to path to import from generate_GReX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_GReX import load_model_from_checkpoint, generate_grex_code

def run_command(command, cwd=None):
    """Run a shell command using os.system"""
    current_dir = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        
        # Run the command and get the return code
        return_code = os.system(command)
        
        # os.system returns exit status, 0 means success
        return_status = return_code == 0
        return return_status
    finally:
        # Change back to the original directory
        if cwd:
            os.chdir(current_dir)

def check_compilation(grex_dir):
    """Check if compilation was successful"""
    print("Compiling GReX code...")
    success = run_command("rm -rf log.txt && make -j128 > log.txt 2>&1", cwd=grex_dir)
    
    # Check if compilation was successful by looking for "SUCCESS" in the last lines of log.txt
    log_path = os.path.join(grex_dir, "log.txt")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_lines = f.readlines()
            last_lines = ''.join(log_lines[-5:])  # Check last 5 lines
            if "SUCCESS" in last_lines:
                print("Compilation successful!")
                return True
    
    print("Compilation failed.")
    return False

def run_grex_simulation(run_dir, executable_path):
    """Run the GReX simulation"""
    print("Running GReX simulation...")
    
    # Set environment variables and run the simulation
    cmd = """
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
export AMREX_CUDA_ARCH=8.0
export MPICH_GPU_SUPPORT_ENABLED=0
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
rm -r plt* 2>/dev/null
rm -r *old* 2>/dev/null
srun -n 128 --ntasks-per-node=32 --gpu-bind=none ./main2d.gnu.TPROF.MPI.OMP.CUDA.ex inputs > log.txt 2>&1
"""
    run_command(cmd, cwd=run_dir)
    
    # Check if plt files were generated regardless of command return status
    # since we care about output files, not just exit code
    plt_files = glob.glob(os.path.join(run_dir, "plt0000[1-5]"))
    if plt_files:
        print(f"Simulation successful! Generated {len(plt_files)} plt files.")
        return True, len(plt_files)
    
    print("Simulation failed or didn't produce expected output files.")
    return False, 0

def calculate_bleu_score(reference_path, generated_path):
    """Calculate BLEU score between reference and generated files"""
    try:
        with open(reference_path, 'r') as f:
            reference = f.read().split()
        with open(generated_path, 'r') as f:
            generated = f.read().split()
        
        # Calculate BLEU score
        smoothie = SmoothingFunction().method1
        score = sentence_bleu([reference], generated, smoothing_function=smoothie)
        return score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def compare_rho_energy(gen_plt_path, gt_plt_path, extent=20):
    """
    Compare RHO_ENERGY data between generated and ground truth plt files
    
    Args:
        gen_plt_path: Path to the generated plt file
        gt_plt_path: Path to the ground truth plt file
        extent: The extent of the xy plane to compare (from -extent to +extent)
        
    Returns:
        float: Similarity score between 0 and 1, where 1 means identical
    """
    if yt is None:
        print("Skipping RHO_ENERGY comparison: yt package not available")
        return 0.0
    
    try:
        print(f"Comparing RHO_ENERGY data between {gen_plt_path} and {gt_plt_path}")
        
        # Load datasets
        gen_ds = yt.load(gen_plt_path)
        gt_ds = yt.load(gt_plt_path)
        
        # Set extent
        extent_x = extent
        extent_y = extent
        
        # Extract RHO_ENERGY from generated simulation
        gen_slc = yt.SlicePlot(gen_ds, 'z', 'RHO_ENERGY')
        gen_slc.set_width((extent_x, extent_y))
        gen_frb = gen_slc.frb
        gen_data = gen_frb['RHO_ENERGY'].d
        
        # Extract RHO_ENERGY from ground truth
        gt_slc = yt.SlicePlot(gt_ds, 'z', 'RHO_ENERGY')
        gt_slc.set_width((extent_x, extent_y))
        gt_frb = gt_slc.frb # fixed resolution buffer
        gt_data = gt_frb['RHO_ENERGY'].d
        
        # Calculate difference
        abs_diff = np.sum(np.abs(gen_data - gt_data))
        gt_sum = np.sum(np.abs(gt_data))
        
        print(f"abs_diff: {abs_diff}, gt_sum: {gt_sum}")
        # Avoid division by zero
        if gt_sum == 0:
            return 0.0
        
        # Check if abs_diff is finite
        if not np.isfinite(abs_diff):
            return 0.0
        
        # Calculate similarity score (1 - normalized difference)
        # Clip to ensure score is between 0 and 1
        similarity = max(0.0, min(1.0, 1.0 - (abs_diff / gt_sum)))
        
        print(f"RHO_ENERGY similarity score: {similarity:.4f}")
        return similarity
        
    except Exception as e:
        print(f"Error comparing RHO_ENERGY data: {e}")
        return 0.0

def evaluate_grex(checkpoint_path, missing_lines_start, missing_lines_end):
    """Evaluate the GReX code generated by the model"""
    # Paths
    grex_dir = "/pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval"
    run_dir = "/pscratch/sd/x/xinshuo/runGReX/20250707_eval"
    template_path = f"{grex_dir}/gt/id.cpp"
    output_path = f"{grex_dir}/id.cpp"
    executable = f"{grex_dir}/main2d.gnu.TPROF.MPI.OMP.CUDA.ex"
    run_executable = f"{run_dir}/main2d.gnu.TPROF.MPI.OMP.CUDA.ex"
    
    # Results dictionary
    results = {
        "compilation_success": False,
        "simulation_success": False,
        "plt_files_count": 0,
        "header_bleu_score": 0.0,
        "rho_energy_similarity": 0.0,
        "total_score": 0.0
    }
    
    # Step 1: Load model and generate code
    try:
        trainer = load_model_from_checkpoint(checkpoint_path)
        generate_grex_code(trainer, template_path, output_path, missing_lines_start, missing_lines_end)
        
        # Free GPU memory after generation
        print("Freeing GPU memory...")
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved() / (1024**2):.2f} MB reserved")
    except Exception as e:
        print(f"Error generating code: {e}")
        return results
    
    # Step 2: Compile the code
    results["compilation_success"] = check_compilation(grex_dir)
    if not results["compilation_success"]:
        return results
    
    # Step 3: Copy executable to run directory
    try:
        if os.path.exists(run_executable):
            os.remove(run_executable)
        shutil.copy(executable, run_executable)
        print(f"Copied executable to {run_executable}")
    except Exception as e:
        print(f"Error copying executable: {e}")
        return results
    
    # Step 4: Run the simulation
    results["simulation_success"], results["plt_files_count"] = run_grex_simulation(run_dir, run_executable)
    if not results["simulation_success"]:
        return results
    
    # Step 5: Compare headers with ground truth
    for i in range(1, 6):
        gen_header = f"{run_dir}/plt0000{i}/Header"
        gt_header = f"{run_dir}/gt/plt0000{i}/Header"
        
        if os.path.exists(gen_header) and os.path.exists(gt_header):
            bleu = calculate_bleu_score(gt_header, gen_header)
            print(f"BLEU score for plt0000{i}/Header: {bleu:.4f}")
            results["header_bleu_score"] += bleu
    
    # Average BLEU score
    if results["plt_files_count"] > 0:
        results["header_bleu_score"] /= results["plt_files_count"]
    
    # Step 6: Compare RHO_ENERGY data for plt00004
    gen_plt = f"{run_dir}/plt00004"
    gt_plt = f"{run_dir}/gt/plt00004"
    
    if os.path.exists(gen_plt) and os.path.exists(gt_plt):
        results["rho_energy_similarity"] = compare_rho_energy(gen_plt, gt_plt, extent=40)
    else:
        print(f"Cannot compare RHO_ENERGY: missing plt00004 files")
    
    # Calculate total score (compilation + simulation + BLEU + RHO_ENERGY)
    results["total_score"] = (
        0.25 * float(results["compilation_success"]) +
        0.25 * float(results["simulation_success"]) +
        0.25 * results["header_bleu_score"] +
        0.25 * results["rho_energy_similarity"]
    )
    
    print(f"Evaluation complete! Total score: {results['total_score']:.4f}")
    return results

def main():
    """Main function to evaluate GReX code generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GReX code generation")
    parser.add_argument("--checkpoint", type=str, default="/pscratch/sd/x/xinshuo/takehome/checkpoints/checkpoint-e0-b10.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--start_line", type=int, required=True,
                        help="Starting line number of the section to replace")
    parser.add_argument("--end_line", type=int, required=True,
                        help="Ending line number of the section to replace")
    
    args = parser.parse_args()
    
    print(f"Evaluating model from checkpoint: {args.checkpoint}")
    print(f"Replacing lines {args.start_line}-{args.end_line}")
    
    # Run evaluation
    results = evaluate_grex(args.checkpoint, args.start_line, args.end_line)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Compilation Success: {results['compilation_success']}")
    print(f"Simulation Success: {results['simulation_success']}")
    print(f"PLT Files Generated: {results['plt_files_count']}")
    print(f"Header BLEU Score: {results['header_bleu_score']:.4f}")
    print(f"RHO_ENERGY Similarity: {results['rho_energy_similarity']:.4f}")
    print(f"Total Score: {results['total_score']:.4f}")

if __name__ == "__main__":
    main()




