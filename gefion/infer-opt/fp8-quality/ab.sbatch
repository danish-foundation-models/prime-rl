#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=fp8q-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=02:30:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/fp8-quality/results/ab-%x-%j.out
# FP8-vs-bf16 sampling-quality A/B on one node, TP8+EP8, no PD (the live run's config otherwise).
#   sbatch ab.sbatch <arm> [stage]
#   arm:   bf16 | fp8 | fp8keep (fp8 with NVIDIA's bf16 keep-list: attention qkv/o + MoE latent projections)
#          | nvidia (NVIDIA's post-trained Nemotron-3 Super, bf16, same config: checkpoint comparison)
#   stage: comma list of score | gen | variants | ckpt | xscore; all = score,gen (default all)
#          variants: sampling variants on this arm (T=0.8, min_p=0.02)
#          ckpt:     8 seeded T=1 resamples of the 387 stratified contexts in data/ckptset.jsonl (token ids kept)
#          xscore:   score the *other* checkpoint arm's ckpt samples (bf16 <-> nvidia, 2 per context) on this
#                    server; waits up to 45 min for the other arm's gen_ckpt.done
# score: teacher-forced logprobs of the 273 normal/degen turns (k=1 all, k=20 on turns <= 32k tokens)
# gen:   8 seeded resamples of 276 contexts (150 that degenerated, 150 normal), live sampling params
# Outputs go to results/<arm>/ (resumable: rerunning skips finished turns/samples).
ARM=${1:?arm: bf16|fp8|fp8keep|nvidia}; STAGE=${2:-all}
[ "$STAGE" = all ] && STAGE=score,gen
has() { [[ ",$STAGE," == *",$1,"* ]]; }
Q=/dcai/projects/iu_0145/rasmus/infer-opt/fp8-quality
OUT=$Q/results/$ARM
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh   # venv, CUDA, caches, stop_server/gpus_free
ulimit -n $(ulimit -Hn)
SET=$Q/data/abset.jsonl
MP=/dcai/projects/iu_0145/rasmus/models/nemotron-super-r6-step2638

case $ARM in
  bf16)    PREC=bf16 ;;
  fp8)     PREC=fp8 ;;
  fp8keep) PREC=fp8 ;;
  nvidia)  PREC=bf16; MP=/dcai/projects/iu_0145/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
# max_num_batched_tokens 4096 and gmu 0.85: prompt_logprobs materialises fp32 logits for every
# scheduled prompt token (4096 x 131072 x 4 B = 2 GiB, plus log-softmax/top-k copies), which the
# memory profiler does not budget for. Same values in every arm.
$PY -I $Q/make_cfg.py $OUT/inference.json $PREC 4096 0.85 $MP >/dev/null || exit 1
if [ $ARM = fp8keep ]; then
  $PY -I - $OUT/inference.json <<'PYEOF'
import json, sys
f = sys.argv[1]; d = json.load(open(f))
d["vllm"]["quantization_config"] = {"ignore": [
    r"re:.*\.(qkv_proj|q_proj|k_proj|v_proj|o_proj)$",       # attention
    r"re:.*\.(fc1_latent_proj|fc2_latent_proj)$"]}            # MoE latent projections
json.dump(d, open(f, "w"), indent=1)
PYEOF
fi

echo "===== $ARM ($STAGE)  $(date -Is)"; t0=$(date +%s)
inference @ $OUT/inference.json > $OUT/server.log 2>&1 &
SRV_PID=$!
until curl -sf localhost:8000/health >/dev/null 2>&1 || ! kill -0 $SRV_PID 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 1500 ]; do sleep 10; done
if ! curl -sf localhost:8000/health >/dev/null 2>&1; then
  echo "server FAILED to start"; grep -E "Error|error" $OUT/server.log | grep -v warn | tail -8 | cut -c1-400; stop_server; exit 1
fi
echo "server ready after $(( $(date +%s) - t0 )) s"
grep -m4 -oE "Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Using [A-Za-z_ ]+ (Unquantized|Fp8)[A-Za-z ]* backend" $OUT/server.log | sort -u
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2

if has score; then
  t1=$(date +%s)
  $PY -I $Q/score.py --model $MP --set $SET --out $OUT/score_k1.jsonl --k 1 --conc 8 || { stop_server; exit 1; }
  $PY -I $Q/score.py --model $MP --set $SET --out $OUT/score_k20.jsonl --k 20 --max-len 32768 --conc 4 || { stop_server; exit 1; }
  echo "score stage $(( $(date +%s) - t1 )) s"
fi
if has gen; then
  t1=$(date +%s)
  timeout 5400 $PY -I $Q/gen_ab.py --model $MP --set $SET --out $OUT/gen.jsonl --n 8 --conc 256
  echo "gen stage $(( $(date +%s) - t1 )) s (exit $?)"
  grep -oE "Avg generation throughput: [0-9.]+ tokens/s" $OUT/server.log | sort -t: -k2 -n | tail -1
fi
if has variants; then
  for v in "t08|--temperature 0.8" "minp02|--min-p 0.02"; do
    t1=$(date +%s)
    timeout 3600 $PY -I $Q/gen_ab.py --model $MP --set $SET --out $OUT/gen_${v%%|*}.jsonl --n 8 --conc 256 ${v#*|}
    echo "variant ${v%%|*} $(( $(date +%s) - t1 )) s"
  done
fi
if has ckpt; then
  t1=$(date +%s)
  timeout 5400 $PY -I $Q/gen_ab.py --model $MP --set $Q/data/ckptset.jsonl --roles ck_turn0,ck_lt8k,ck_8_32k,ck_ge32k \
    --out $OUT/gen_ckpt.jsonl --n 8 --conc 256 --save-ids; touch $OUT/gen_ckpt.done  # also after a timeout: partial samples are usable
  echo "ckpt gen $(( $(date +%s) - t1 )) s"
fi
if has xscore; then
  case $ARM in bf16) OTHER=nvidia ;; nvidia) OTHER=bf16 ;; *) OTHER= ;; esac
  if [ -n "$OTHER" ]; then
    for i in $(seq 1 270); do [ -f $Q/results/$OTHER/gen_ckpt.done ] && break; sleep 10; done
    t1=$(date +%s)
    $PY -I $Q/score.py --model $MP --set $Q/data/ckptset.jsonl --roles ck_turn0,ck_lt8k,ck_8_32k,ck_ge32k \
      --gen $Q/results/$OTHER/gen_ckpt.jsonl --per-context 2 --out $OUT/xscore.jsonl --k 1 --conc 8
    echo "xscore of $OTHER samples $(( $(date +%s) - t1 )) s"
  fi
fi
stop_server
echo "done $(date -Is), total $(( $(date +%s) - t0 )) s"
