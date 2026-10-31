#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-mtpfix-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=00:45:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtpfix-g-%j.out
# MTP + mamba "align" mode corrupts decoding from position 4096 (= max_num_batched_tokens) to the end of the first cache block
# (4176, or 4224 with block_size=4224: jobs 531183, 531207). Test the max_num_batched_tokens link:
# (a) max_num_batched_tokens = block size 4176 (empty window?), (b) 8192 (does the window move to 8192-8352?).
# 32 i3-math prompts, forced length.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtpfix-g-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export INFER_OPT=ep_compact_moe,fast_generate VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs
PROMPTS=$IO/bench/data/i3math_prompts.json
MTP='"vllm.gpu_memory_utilization": 0.91, "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}'

for v in "mtp1_mnbt4176|{$MTP, \"vllm.max_num_batched_tokens\": 4176}|4800" "mtp1_mnbt8192|{$MTP, \"vllm.max_num_batched_tokens\": 8192}|8800"; do
  IFS="|" read -r name ov maxt <<< "$v"
  if start_server $name "$ov"; then
    grep -m2 -oE "Setting attention block size to [0-9]+|Padding mamba page size by [0-9.]+%" $OUT/$name.log | sort -u
    grep -m1 -oE "block_size=[0-9]+|'block_size': [0-9]+" $OUT/$name.log | head -1
    $PY -I $IO/bench/parity.py gen --model "$MODEL" --prompts-file $PROMPTS --n-prompts 32 --reps 1 --force-length \
      --max-tokens $maxt --out $OUT/fix-$name.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/fix-$name.json --out $OUT/fix-$name-tf.json 2>&1 | tail -1
    $PY -I $IO/analysis/mismatch_bursts.py $OUT/fix-$name.json $OUT/fix-$name-tf.json 2>&1 | tail -4
    $PY -I $IO/bench/parity.py compare --b $OUT/fix-$name.json --b-self-tf $OUT/fix-$name-tf.json \
      --out $OUT/fix-parity-$name.json 2>&1 | tail -1 | cut -c1-200
  fi
  stop_server
done
echo "end=$(date -Is)"
