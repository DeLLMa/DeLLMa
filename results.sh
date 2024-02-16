echo "Running DeLLMa-Pairs on Agriculture"
python evaluate_dellma.py --pref_enum_mode rank-minibatch --sample_size 64 --alpha 2e-3

echo "Running DeLLMa-Top1 on Agriculture"
python evaluate_dellma.py --pref_enum_mode rank-minibatch --sample_size 64 --mode top1 --alpha 1e-3

echo "Running DeLLMa-Naive on Agriculture"
python evaluate_dellma.py --pref_enum_mode rank --sample_size 50 --alpha 1e-3

echo "Running DeLLMa-Pairs on Stocks"
python evaluate_dellma.py --agent_name trader --softmax_mode action --overlap_pct 0.5 --pref_enum_mode rank-minibatch --sample_size 64 --alpha 2e-8 --temperature 0.1

echo "Running DeLLMa-Top1 on Stocks"
python evaluate_dellma.py --agent_name trader --softmax_mode action --overlap_pct 0.5 --pref_enum_mode rank-minibatch --sample_size 64 --mode top1 --alpha 2e-8 --temperature 0.1

echo "Running DeLLMa-Naive on Stocks"
python evaluate_dellma.py --agent_name trader --softmax_mode action --pref_enum_mode rank --sample_size 50  --alpha 2e-8

echo "Baselines..."

echo "Running Zero-Shot on Agriculture"
python evaluate_dellma.py --pref_enum_mode zero-shot

echo "Running CoT on Agriculture"
python evaluate_dellma.py --pref_enum_mode cot

echo "Running Self-Consistency on Agriculture"
python evaluate_dellma.py --pref_enum_mode self-consistency

echo "Running Zero-Shot on Stocks"
python evaluate_dellma.py --agent_name trader --pref_enum_mode zero-shot

echo "Running CoT on Stocks"
python evaluate_dellma.py --agent_name trader --pref_enum_mode cot

echo "Running Self-Consistency on Stocks"
python evaluate_dellma.py --agent_name trader --pref_enum_mode self-consistency
