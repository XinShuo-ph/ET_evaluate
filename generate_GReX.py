# load the checkpoints in /pscratch/sd/x/xinshuo/takehome/checkpoints/checkpoint-e0-b10.pt
# as a PPOTrainer defined in ppo_trainer.py

# then write prompt for the LLM to generate the GReX code

# let's aim to generate lines in /pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval/gt/id.cpp
# specifically, let's use the id.cpp file (excluding one line, or a few lines) as context
# write a prompt for the LLM to generate the missing line (or a few lines)
# then replace the corresponding lines in /pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval/id.cpp

import os
import torch
import sys
import shutil
import gc

# Add parent directory to path to import PPOTrainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo_trainer import PPOTrainer, create_structured_prompt

def load_model_from_checkpoint(checkpoint_path):
    """Load a trained model from checkpoint"""
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize trainer with same model as in checkpoint
    trainer = PPOTrainer(
        model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",  # Default model, will be overridden by checkpoint
        learning_rate=1e-7,
        clip_epsilon=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        bleu_weight=0.8,
        llm_weight=0.2,
        use_llm_judge=False,
        generation_config={
            "max_new_tokens": 400,
            "temperature": 0.3,
            "do_sample": True
        }
    )
    
    # Load model weights from checkpoint
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.value_net.load_state_dict(checkpoint['value_net_state_dict'])
    
    print(f"Model loaded from checkpoint {checkpoint_path}")
    return trainer

def generate_grex_code(trainer, template_path, output_path, missing_lines_start, missing_lines_end):
    """Generate GReX code using the trained model"""
    # Read the template file
    with open(template_path, 'r') as f:
        template_code = f.read()
    
    lines = template_code.split('\n')
    
    # Extract lines before and after the missing section
    before_missing = '\n'.join(lines[:missing_lines_start])
    after_missing = '\n'.join(lines[missing_lines_end:])
    
    # Extract file name from path
    file_name = os.path.basename(template_path)
    
    # Create structured prompt with context
    prompt = f"""<|system|>
You are an expert C++ developer. You excel at implementing complex physics algorithms and writing efficient, correct code.

<|user|>
I need you to complete {missing_lines_end - missing_lines_start} lines of code in the file `{file_name}`.


## Code Context
The code before the missing section:
```cpp
{before_missing}
```

The code after the missing section:
```cpp
{after_missing}
```

Please generate only the missing {missing_lines_end - missing_lines_start} lines of code that should go between these two parts. Follow the existing coding style and ensure the implementation is correct and compatible with the AMREX framework.

<|assistant|>
Here's the {missing_lines_end - missing_lines_start} lines of code for the missing section:
```cpp"""
    
    # Generate code
    generated_code, _ = trainer.generate_response(prompt)

    print(f"Generated code: {generated_code}")
    
    # Clean up the generated code using the approach from load_infer.py
    generated_code = generated_code.strip()
    
    # if the model decide to complement with ```, remove it and any supplementary text after
    if "```" in generated_code:
        generated_code = generated_code.split("```")[0].strip()
    
    
    final_code = generated_code.strip()
    
    # Log the generated code
    print(f"Generated code: {final_code}")
    
    # If we're filling in a specific section
    if missing_lines_start is not None and missing_lines_end is not None:
        result_lines = lines.copy()
        # Replace the missing section with generated code
        result_lines[missing_lines_start:missing_lines_end] = final_code.split('\n')
        final_code = '\n'.join(result_lines)
    
    # Write the generated code to the output file
    with open(output_path, 'w') as f:
        f.write(final_code)
    
    print(f"Generated code written to {output_path}")
    return final_code

def main():
    """Main function to generate GReX code from a checkpoint"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GReX code using a trained model")
    parser.add_argument("--checkpoint", type=str, default="/pscratch/sd/x/xinshuo/takehome/checkpoints/checkpoint-e0-b0.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--template", type=str, default="/pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval/gt/id.cpp",
                        help="Path to the template file")
    parser.add_argument("--output", type=str, default="/pscratch/sd/x/xinshuo/GReX/Problems/FUKA_BBH_ScalarField_boost_r35_box6400_for_eval/id.cpp",
                        help="Path where the generated code will be saved")
    parser.add_argument("--start_line", type=int, required=True,
                        help="Starting line number of the section to replace")
    parser.add_argument("--end_line", type=int, required=True,
                        help="Ending line number of the section to replace")
    
    args = parser.parse_args()
    
    # Load model
    trainer = load_model_from_checkpoint(args.checkpoint)
    
    # Generate code
    print(f"Generating code to replace lines {args.start_line}-{args.end_line}")
    print(f"Template: {args.template}")
    print(f"Output: {args.output}")
    
    generate_grex_code(trainer, args.template, args.output, args.start_line, args.end_line)
    
    # Free GPU memory after generation
    print("Freeing GPU memory...")
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved() / (1024**2):.2f} MB reserved")

if __name__ == "__main__":
    main()

