#!/bin/bash
# PD variant of order_nodes.sh: print the job's hostnames with the iommu=pt nodes FIRST, so the
# launcher (which gives the first NUM_INFER_NODES hostnames to inference; with PD the prefill node(s)
# come first, then the decode node(s)) puts the prefill/decode pair on fixed nodes. The NIXL KV
# transfer between them is GPUDirect RDMA, which runs 4-7x slow without iommu=pt. Remaining fixed
# nodes, then the unfixed ones, go to the trainer.
# Fails if fewer than NUM_INFER_NODES + NUM_TRAIN_NODES - MAX_OFFPOOL_TRAIN_NODES nodes are fixed, except
# for the shakedown job (job name gefion-super-shakedown), which continues with whatever it got.
# Unfixed nodes go last, so they become trainer nodes and never trainer rank 0 (the weight broadcast host).
set -uo pipefail
# One trainer node may lack iommu=pt (user decision 2026-11-04, when other users held two pool nodes).
# Its FSDP and broadcast NCCL traffic may be slower; the trainer is idle for ~2/3 of each step.
MAX_OFFPOOL_TRAIN_NODES=${MAX_OFFPOOL_TRAIN_NODES:-1}
types=$(srun --ntasks-per-node=1 --overlap bash -c '
  pci=$(readlink -f /sys/class/infiniband/mlx5_0/device)
  echo "$(hostname -s) $(cat $pci/iommu_group/type)"') || { echo "order_nodes_pd: probe failed" >&2; exit 1; }
good=(); bad=()
for h in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  t=$(awk -v h="$h" '$1 == h {print $2}' <<<"$types")
  if [ "$t" = identity ]; then good+=("$h"); else bad+=("$h"); fi
done
echo "order_nodes_pd: iommu=pt (inference first): ${good[*]:-none}; without: ${bad[*]:-none}" >&2
if [ "${#good[@]}" -lt "$((NUM_INFER_NODES + NUM_TRAIN_NODES - MAX_OFFPOOL_TRAIN_NODES))" ]; then
  echo "order_nodes_pd: only ${#good[@]} nodes have iommu=pt, want $((NUM_INFER_NODES + NUM_TRAIN_NODES - MAX_OFFPOOL_TRAIN_NODES))" >&2
  [ "${SLURM_JOB_NAME:-}" = gefion-super-shakedown ] || exit 1
  [ "${#good[@]}" -ge "$NUM_INFER_NODES" ] || echo "order_nodes_pd: WARNING inference pair not all on iommu=pt nodes" >&2
  echo "order_nodes_pd: shakedown job, continuing" >&2
fi
echo "${good[@]}" "${bad[@]}"
