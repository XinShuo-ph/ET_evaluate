from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from bleu_reward import compute_bleu_score
from llm_judge import LLMCodeJudge
from combined_reward import CombinedReward

# we are constrained in computing resources, set upper limits
context_char_limit = 10000
generated_char_limit = 4000
generated_max_tokens = 400

def create_structured_prompt(example, char_limit=context_char_limit):
    """Structured prompt with clear sections"""
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
{example['interface']}
```

## Schedule Definition in schedule.ccl:
```
{example['schedule']}
```

## Parameters Definition in param.ccl:
```
{example['param']}
```

## Configuration Definition in configuration.ccl:
```
{example['configuration']}
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

# Load model
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

# load llm judge
myjudge = LLMCodeJudge()

# load combined reward
reward_system = CombinedReward()

# Load dataset
ds = load_dataset("xinshuo/ET_code_with_context")
train_ds = ds['train']

# Test on examples
test_indices = [1, 3, 5]
for i in test_indices:
    print(f"\n{'='*60}")
    print(f"Example {i+1}: {train_ds[i]['thorn_name']} - {train_ds[i]['src_filename']}")
    print('='*60)
    
    example = train_ds[i]

    prompt = create_structured_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generated_max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print("decoding...")

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_code = generated[len(prompt):].strip()
    
    # Clean up artifacts
    # if the model decide to complement with ```, remove it and any supplementary text after
    if "```" in generated_code:
        generated_code = generated_code.split("```")[0].strip()
    # since our prompt ends with ```, we need to remove the things before the first \n, which likely is just the language tag
    if "\n" in generated_code and generated_code.find("\n") < 50:
        first_newline = generated_code.find("\n")
        generated_code = generated_code[first_newline+1:]
    

    gt_code = train_ds[i]['src_code']
    print(f"Generated code ({len(generated_code)} chars):")
    print(generated_code[:100] + "\n...(more code truncated)" if len(generated_code) > 100 else generated_code)
    
    print(f"\nActual code ({len(gt_code)} chars):")
    print(gt_code[:100] + "\n...(more code truncated)" if len(gt_code) > 100 else gt_code)

    # Compute and print BLEU score, constrain the length of gt_code
    gt_code = gt_code[:generated_char_limit]
    bleu_score = compute_bleu_score(generated_code, gt_code)
    print(f"\nBLEU Score: {bleu_score:.4f}")

    # Compute and print LLM judge score
    llm_judge_score = myjudge.judge_code(generated_code, gt_code)
    print(f"\nLLM Judge Score: {llm_judge_score['llm_judge_score']:.4f}")
    # print(f"LLM Judge Raw Response: {llm_judge_score['raw_response']}")
    # print(f"LLM Judge Score Confidence: {llm_judge_score['score_confidence']:.4f}")

    # Combined reward
    combined_reward = reward_system.get_reward(generated_code, gt_code)
    print(f"\nCombined Reward: {combined_reward['combined_reward']:.4f}")
