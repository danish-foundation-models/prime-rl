# Shared helpers for single-node infer-opt jobs. Source from an sbatch script after setting OUT.
# Variables live in scripts: the site cancels jobs submitted with --export=VAR=...
set -uo pipefail
IO=/dcai/projects/iu_0145/rasmus/infer-opt
S=/dcai/projects/iu_0145/rasmus/phase0/smoke
PR=/dcai/projects/iu_0145/rasmus/phase0/prime-rl
PY=$PR/.venv/bin/python
MODEL=/dcai/projects/iu_0145/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
NSYS=/dcai/software/tenant/apps/CUDA/13.0.0/bin/nsys
PYSPY=$IO/tools/pydeps/bin/py-spy
cd $PR
export CUDA_HOME=/dcai/software/tenant/apps/CUDA/13.0.0  # nvcc for JIT kernels
export PATH=$PATH:$CUDA_HOME/bin
source .venv/bin/activate
export PATH=$PATH:/dcai/projects/iu_0145/rasmus/phase0/uvenv/bin UV_NO_SYNC=1 UV_FROZEN=1
export HF_HOME=/dcai/projects/iu_0145/rasmus/.cache/huggingface
# Per-job compile caches on local NVMe. A shared vLLM compile cache that points into another job's
# Triton dir is what broke DP8+FP8 in jobs 528375/528393 ("CUDA driver error: file not found").
export VLLM_CACHE_ROOT=/tmp/$SLURM_JOB_ID/vllm-cache TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton-cache
mkdir -p $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $OUT
# nsys flushes stall every worker at once; don't let vLLM declare the engine dead meanwhile.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export PYTHONPATH=$IO${PYTHONPATH:+:$PYTHONPATH}  # infer_opt patches via sitecustomize (INFER_OPT=...)
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1

SRV_PID=""
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)" = 0 ]; }

# start_server <name> <overrides-json> [nsys]   (INFER_OPT env selects patches)
start_server() {
  local name=$1 ov=$2 mode=${3:-} cfg log t0
  cfg=$($PY -I $IO/bench/make_config.py $OUT/$name.json "$ov") || return 1
  log=$OUT/$name.log
  echo; echo "===== $name  $(date -Is)  INFER_OPT=${INFER_OPT:-}  overrides=$ov"
  t0=$(date +%s)
  if [ "$mode" = nsys ]; then
    $NSYS launch --session-new=io$SLURM_JOB_ID --trace=cuda,nvtx --cuda-graph-trace=node --wait=all \
      inference @ "$cfg" > "$log" 2>&1 &
  else
    inference @ "$cfg" > "$log" 2>&1 &
  fi
  SRV_PID=$!
  until curl -sf localhost:8000/health >/dev/null 2>&1 || ! kill -0 $SRV_PID 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 1200 ]; do sleep 10; done
  if curl -sf localhost:8000/health >/dev/null 2>&1; then
    echo "$name ready after $(( $(date +%s) - t0 )) s"
    grep -m3 -oE "Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens.*|Using [A-Za-z_ ]+ (Unquantized|Fp8) MoE backend" "$log" | sort -u
    return 0
  fi
  echo "$name FAILED to start"; grep -E "Error|error" "$log" | grep -vE "warn|Configuration class" | tail -6 | cut -c1-400
  stop_server; return 1
}

stop_server() {
  [ -n "$SRV_PID" ] && kill $SRV_PID 2>/dev/null; sleep 10; pkill -9 -f "inference @|vllm|VLLM" 2>/dev/null
  for i in $(seq 1 30); do gpus_free && break; sleep 5; done
  # the next server must be able to bind :8000 (opt1 "compact" failed with EADDRINUSE)
  for i in $(seq 1 24); do ss -ltnH 'sport = :8000' | grep -q . || break
    [ $i = 6 ] && for p in $(ss -ltnpH 'sport = :8000' | grep -oE 'pid=[0-9]+' | cut -d= -f2); do kill -9 $p; done
    sleep 5; done
  SRV_PID=""
}

nsys_capture() {  # nsys_capture <file-stem> <seconds>
  $NSYS start --session=io$SLURM_JOB_ID --sample=none --cpuctxsw=none -f true -o $OUT/$1 && sleep $2 && \
    $NSYS stop --session=io$SLURM_JOB_ID && echo "nsys capture $1 done"
}

cpu_snapshot() {  # top CPU consumers (API server, engine cores, workers) during load
  ps -eo pid,pcpu,rss,args --sort=-pcpu | grep -E "python|VLLM|inference" | grep -v grep | head -14 | cut -c1-160
}

synthetic() {  # synthetic <tag>: the section-11 load (1.5K prompts, 2048 forced tokens, 768 concurrent)
  echo -n "$1 synthetic "; timeout 900 $PY -I $S/bench_client.py http://localhost:8000/v1 "$MODEL" 768 2048 2>&1 | tail -1
  grep -oE "Avg generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs" $OUT/$1.log | sort -t: -k2 -n | tail -1 | sed "s/^/$1 peak engine: /"
}

rl() {  # rl <tag> <extra rl_load args...>
  local tag=$1; shift
  timeout 1500 $PY -I $IO/bench/rl_load.py --model "$MODEL" --tag "$tag" "$@" 2>&1 | grep --line-buffered -E "RESULT|check:|error|HTTP|retry" | head -12
}

parity_run() {  # parity_run <tag>: seeded gen + teacher-forced scores of own and reference seqs
  # PARITY_REF: directory holding par-ref.json / par-ref-tf.json (default: this job's OUT)
  local tag=$1 ref=${PARITY_REF:-$OUT} pa="--reps 1 --max-tokens 768"
  $PY -I $IO/bench/parity.py gen --model "$MODEL" $pa --out $OUT/par-$tag.json 2>&1 | tail -1
  $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/par-$tag.json --out $OUT/par-$tag-tf.json 2>&1 | tail -1
  if [ "$tag" != ref ]; then
    for i in $(seq 1 90); do [ -f $ref/par-ref-tf.json ] && break; sleep 10; done
    $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $ref/par-ref.json --out $OUT/par-$tag-tf-on-ref.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py compare --a $ref/par-ref.json --b $OUT/par-$tag.json --a-tf $ref/par-ref-tf.json \
      --b-tf $OUT/par-$tag-tf-on-ref.json --b-self-tf $OUT/par-$tag-tf.json --out $OUT/parity-$tag.json 2>&1 | tail -1
  fi
}
