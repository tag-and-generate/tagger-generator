###########
# Usage: bash eval/run_context_eval.sh <hypothesis_filepath> <reference_filepath>
###########

export PYTHONPATH='eval/nlg_eval:.'
hyp=$1
ref=$2

python3 eval/context_eval.py --hyp "$1" --ref "$2"
tail -n 1 <(cat $hyp | sacrebleu -w 2 $ref)
