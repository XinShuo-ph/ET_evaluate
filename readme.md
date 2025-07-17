
`bleu_reward.py`, `llm_judge.py` and `compile_reward.py` define the reward structure for the task of generating code for the Einstein Toolkit: combination of BLEU score and LLM judge grading.

Running `python load_infer.py > log.txt 2>&1` will generate a few examples.

## Evaluation Metrics
`generate_GReX.py`, `evaluate_GReX.py` and `evaluate_all_checkpoints.py` are the code for evaluating the model on the GReX code. Running `python evaluate_all_checkpoints.py --start_line 376 --end_line 378 --batch_range 0,180,10 --output_dir ./evaluation_results_376_378` generates the data in `evaluation_results_376_378`.

## Training
`ppo_trainer.py` and `train_rl.py` defines the training loop. For example

```bash
python train_rl.py --model_name Qwen/Qwen2.5-Coder-0.5B-Instruct\
          --num_epochs 5 \
          --batch_size 3 \
          --ppo_epochs 4\
          --learning_rate 1e-7 \
          --clip_epsilon 0.02 \
          --use_llm_judge --use_wandb 
```
