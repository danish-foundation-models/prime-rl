#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-fp8mtp-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=01:45:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/fp8mtp-h-%j.out
# FP8 + MTP with prefix caching on and host-RAM KV offload, on a multi-turn agent-style load.
# All variants: mamba "align" mode, max_num_batched_tokens 4176 (avoids the MTP + align corruption),
# OffloadingConnector with 96 GiB host RAM per DP engine, ep_compact_moe + fast_generate.
# FP8 adds fi_moe_chunk (FlashInfer MoE workspace) and dg_jit_per_proc (DeepGEMM JIT cache race).
# Order: FP8 variants first, then the bf16 control, which also scores the FP8 sequences so FP8's
# sampler logprobs are compared with bf16 teacher forcing (the trainer is bf16).
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/fp8mtp-h-$SLURM_JOB_ID
ulimit -n $(ulimit -Hn)
ulimit -l unlimited 2>/dev/null || true  # pinned host buffers for the offload tier
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs INFER_OPT_DG_CACHE_BASE=/tmp/$SLURM_JOB_ID/dg-cache
PROMPTS=$IO/bench/data/i3math_prompts.json
COMMON='"vllm.gpu_memory_utilization": 0.91, "vllm.enable_prefix_caching": true, "vllm.mamba_cache_mode": "align", "vllm.max_num_batched_tokens": 4176, "kv_cache_offload": {"type": "native", "cpu": {"num_bytes": 103079215104}}'
FP8='"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176'
mtp() { echo "\"vllm.speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": $1}"; }

run() {  # run <name> <INFER_OPT> <overrides> <concurrency>
  local name=$1 conc=$4
  export INFER_OPT=$2
  rm -f /dev/shm/vllm_offload_* 2>/dev/null
  if start_server $name "{$3}"; then
    grep -m4 -oE "Available KV cache memory: [0-9.]+ GiB|Setting attention block size to [0-9]+|Using [A-Z_]+ Fp8 MoE backend|OffloadingConnector[^,]*" $OUT/$name.log | sort -u
    # 8-turn agent-style rollouts: each turn re-sends the history + a synthetic tool result
    rl $name-multiturn --concurrency $conc --warmup 360 --duration 240 --prompts-file $PROMPTS --output 1:4096 \
      --turns 8 --tool-tokens 256:2048 --max-context 65536 --logprobs 33
    grep "Engine 000:" $OUT/$name.log | tail -2 | cut -c40-260
    grep "SpecDecoding metrics" $OUT/$name.log | tail -1 | grep -oE "Mean acceptance length: [0-9.]+"
    grep -m3 -iE "offload.*(hit|load|store)[^ ]* [0-9]" $OUT/$name.log | tail -2 | cut -c40-260
    $PY -I $IO/bench/parity.py gen --model "$MODEL" --prompts-file $PROMPTS --n-prompts 32 --reps 1 \
      --max-tokens 16384 --out $OUT/math-$name.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/math-$name.json --out $OUT/math-$name-tf.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py compare --b $OUT/math-$name.json --b-self-tf $OUT/math-$name-tf.json \
      --out $OUT/math-parity-$name.json 2>&1 | tail -1 | cut -c1-220
    $PY -I $IO/analysis/mismatch_bursts.py $OUT/math-$name.json $OUT/math-$name-tf.json | tail -1
    if [ "$name" = bf16_mtp1 ]; then  # FP8 sampler logprobs vs bf16 teacher forcing
      for f in fp8_mtp1 fp8_mtp2; do
        [ -f $OUT/math-$f.json ] || continue
        $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/math-$f.json --out $OUT/math-$f-tf-bf16.json 2>&1 | tail -1
        $PY -I $IO/bench/parity.py compare --b $OUT/math-$f.json --b-self-tf $OUT/math-$f-tf-bf16.json \
          --out $OUT/math-parity-$f-vs-bf16.json 2>&1 | tail -1 | sed "s/^/$f vs bf16: /" | cut -c1-240
      done
    fi
  fi
  stop_server
}

run fp8_mtp1 ep_compact_moe,fast_generate,fi_moe_chunk,dg_jit_per_proc "$COMMON, $FP8, $(mtp 1)" 1408
run fp8_mtp2 ep_compact_moe,fast_generate,fi_moe_chunk,dg_jit_per_proc "$COMMON, $FP8, $(mtp 2)" 1408
run bf16_mtp1 ep_compact_moe,fast_generate "$COMMON, $(mtp 1)" 768
echo "end=$(date -Is)"
