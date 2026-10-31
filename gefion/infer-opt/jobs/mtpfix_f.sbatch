#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-mtpfix-f
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=00:45:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtpfix-f-%j.out
# MTP + mamba "align" mode corrupts decoding at absolute positions 4097-4176 (job 531183): the tail
# of the first 4176-token cache block, past the last multiple of the 128-token SSD chunk. Test:
# reproduce with the default block size, then set block_size to a chunk multiple (4224 = 33 x 128).
# 32 i3-math prompts, forced to 4800 tokens so every sequence crosses the boundary.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtpfix-f-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export INFER_OPT=ep_compact_moe,fast_generate VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs
PROMPTS=$IO/bench/data/i3math_prompts.json
MTP='"vllm.gpu_memory_utilization": 0.91, "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}'

for v in "mtp1_block4176|{$MTP}" "mtp1_block4224|{$MTP, \"vllm.block_size\": 4224}"; do
  name=${v%%|*}; ov=${v#*|}
  if start_server $name "$ov"; then
    grep -m2 -oE "Setting attention block size to [0-9]+|Padding mamba page size by [0-9.]+%" $OUT/$name.log | sort -u
    grep -m1 -oE "block_size=[0-9]+|'block_size': [0-9]+" $OUT/$name.log | head -1
    $PY -I $IO/bench/parity.py gen --model "$MODEL" --prompts-file $PROMPTS --n-prompts 32 --reps 1 --force-length \
      --max-tokens 4800 --out $OUT/fix-$name.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/fix-$name.json --out $OUT/fix-$name-tf.json 2>&1 | tail -1
    $PY -I $IO/analysis/mismatch_bursts.py $OUT/fix-$name.json $OUT/fix-$name-tf.json 2>&1 | tail -4
    $PY -I $IO/bench/parity.py compare --b $OUT/fix-$name.json --b-self-tf $OUT/fix-$name-tf.json \
      --out $OUT/fix-parity-$name.json 2>&1 | tail -1 | cut -c1-200
  fi
  stop_server
done
echo "end=$(date -Is)"
