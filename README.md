# DeLLMa: A Framework for Decision Making Under Uncertainty with Large Language Models

Reproduce DeLLMa results by running

```
bash ./results.sh
```

## Baslines

* Query GPT-4 for baseline methods. `[AGENT]` can be chosen from `{farmer, trader}`, and `[BASELINE]` can be chosen from `{zero-shot, cot, self-consistency}`

```
python main.py --agent_name [AGENT] --dellma_mode [BASELINE] --results_path PATH/TO/RESULT
```

* Evaluate performance of baseline methods.

```
python evalute_dellma.py --agent_name [AGENT] --pref_enum_mode [BASELINE] --results_path PATH/TO/RESULT
```

## DeLLMa Agents

* Query GPT-4 for DeLLMa-Naive. Here, `[SIZE]` denotes the **total samples size** we use for DeLLMa-Naive (i.e. distributed across all actions). We use 50 in our paper.

```
python main.py --agent_name [AGENT] --dellma_mode rank --sample_size [SIZE] --results_path PATH/TO/RESULT
```

* Query GPT-4 for DeLLMa-{Pairs, Top1}. Here, `[SIZE]` denotes the **per action sample size**. We use 64 for our best performing agent and ablate from 4 to 64 in our ablation studies. `[PCT]` denotes the proportions shared between minibatches. For `farmer`, we use 0.25 and for `trader` we use 0.5. In our ablation studies we study values from 0, 0.25, 0.5, 0.75.

```
python main.py --agent_name [AGENT] --dellma_mode rank-minibatch --sample_size [SIZE] --overlap_pct [PCT] --results_path PATH/TO/RESULT
```

* Evaluate DeLLMa agents. `[DeLLMa]` can be chosen from `{rank, rank-minibatch}`. `[ALPHA]` controls the regularization strength when optimizing the Bradley-Terry Model with the ILSR algorithm, `[SOFTMAX]` governs how we normalize the utility function after the ILSR procedure. Choices from `full` (apply softmax to the complete utility vector) and `action` (first group all state-action pairs by action, then softmax). `[TEMP]` temperature scaling for the softmax.

```
python evaluate_dellma.py    \
    --agent_name [AGENT]     \
    --pref_enum_mode [DeLLMa]  \
    --sample_size [SIZE]     \
    --overlap_pct [PCT]      \
    --alpha [ALPHA]          \
    --softmax_mode [SOFTMAX] \
    --temperature [TEMP]     \
    --results_path PATH/TO/RESULT
```
