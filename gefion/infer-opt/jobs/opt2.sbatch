#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-opt2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:00:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/opt2-%j.out
# Second half of the stack (opt1 has the stock baseline and the parity reference). Each variant: parity vs the
# stock server (decode + prefill + mismatch), the section-11 synthetic load, and the RL-shaped load.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/opt2-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
TUNED=$IO/moe_configs

variant() {  # variant <name> <INFER_OPT> <tuned 0|1> <overrides>
  export INFER_OPT=$2
  if [ "$3" = 1 ]; then export VLLM_TUNED_CONFIG_FOLDER=$TUNED; else unset VLLM_TUNED_CONFIG_FOLDER; fi
  if start_server $1 "$4"; then
    grep -m3 -oE "\[infer_opt[^]]*\] [a-z_]+: .*|ep_compact_moe active.*|Using configuration from .*json" $OUT/$1.log | sort -u
    if [ "$1" = base ]; then
      parity_run ref
      $PY -I $IO/bench/parity.py gen --model "$MODEL" --reps 1 --max-tokens 768 --out $OUT/par-ref2.json | tail -1
      $PY -I $IO/bench/parity.py compare --a $OUT/par-ref.json --b $OUT/par-ref2.json --out $OUT/parity-noise.json | tail -1
    else
      parity_run $1
    fi
    synthetic $1
    rl $1-rl-lp33 --concurrency 768 --warmup 180 --duration 120 --prompt 1024:8192 --output 8192:32768 --logprobs 33
  fi
  stop_server
}

export PARITY_REF=$IO/results/opt1-531096
echo "parity reference: $PARITY_REF"
variant replay "ep_compact_moe,fast_generate" 1 '{"vllm.use_replayssm": true}'
variant replay_fp16 "ep_compact_moe,fast_generate" 1 '{"vllm.use_replayssm": true, "vllm.mamba_ssm_cache_dtype": "float16"}'
variant fp8 "fast_generate" 0 '{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 144}'
echo "end=$(date -Is)"
