# Body of the MTP/FP8 jobs: runs each VARIANTS entry ('name|json overrides[|concurrency]') on one node.
# Order per variant: RL-shaped load first (the production-relevant number), then parity. Parity
# scoring uses prompt_logprobs, which needs ~2 GiB+ of fp32 logits per 4096-token prefill chunk;
# at gpu_memory_utilization 0.93 that OOMed next to the MTP drafter (job 531170), so all variants run at 0.91.
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export INFER_OPT=ep_compact_moe,fast_generate${EXTRA_OPT:+,$EXTRA_OPT} VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs
PROMPTS=$IO/bench/data/i3math_prompts.json
BASE='{"vllm.gpu_memory_utilization": 0.91}'

merge() { $PY -I -c 'import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))' "$BASE" "$1"; }

for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name ovj conc <<< "$v"
  ov=$(merge "$ovj")
  if start_server $name "$ov"; then
    grep -m3 -oE "Available KV cache memory: [0-9.]+ GiB|Using [A-Z_]+ Fp8 MoE backend" $OUT/$name.log | sort -u
    rl $name-math --concurrency ${conc:-768} --warmup 240 --duration 180 --prompts-file $PROMPTS --output 1:16384 --logprobs 33
    grep "Engine 000:" $OUT/$name.log | tail -2 | cut -c40-260
    grep "SpecDecoding metrics" $OUT/$name.log | tail -1 | cut -c40-400
    curl -s -m 600 localhost:8000/health >/dev/null  # let aborted load requests clear before parity
    # natural-length seeded decodes on 32 i3-math prompts, scored teacher-forced on the same server
    $PY -I $IO/bench/parity.py gen --model "$MODEL" --prompts-file $PROMPTS --n-prompts 32 --reps 1 \
      --max-tokens 16384 --out $OUT/math-$name.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/math-$name.json --out $OUT/math-$name-tf.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py compare --b $OUT/math-$name.json --b-self-tf $OUT/math-$name-tf.json \
      --out $OUT/math-parity-$name.json 2>&1 | tail -1
  fi
  stop_server
done
echo "end=$(date -Is)"
