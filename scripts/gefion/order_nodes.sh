#!/bin/bash
# Print the job's hostnames with the nodes lacking iommu=pt first, so the launcher (which gives the
# first NUM_INFER_NODES hostnames to inference) places them on inference and keeps the trainer on
# fixed nodes. Without iommu=pt GPUDirect RDMA runs 4-7x slow, which matters for the trainer's
# collectives; inference only crosses IB for the per-step weight broadcast.
# Fails if fewer than NUM_TRAIN_NODES nodes are fixed.
set -uo pipefail
types=$(srun --ntasks-per-node=1 --overlap bash -c '
  pci=$(readlink -f /sys/class/infiniband/mlx5_0/device)
  echo "$(hostname -s) $(cat $pci/iommu_group/type)"') || { echo "order_nodes: probe failed" >&2; exit 1; }
good=(); bad=()
for h in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  t=$(awk -v h="$h" '$1 == h {print $2}' <<<"$types")
  if [ "$t" = identity ]; then good+=("$h"); else bad+=("$h"); fi
done
echo "order_nodes: iommu=pt: ${good[*]}; without (to inference): ${bad[*]:-none}" >&2
if [ "${#good[@]}" -lt "$NUM_TRAIN_NODES" ]; then
  echo "order_nodes: only ${#good[@]} nodes have iommu=pt, need $NUM_TRAIN_NODES for the trainer" >&2
  # The shakedown only has to work, not be fast: run the trainer partly on unfixed nodes.
  [ "${SLURM_JOB_NAME:-}" = gefion-super-shakedown ] || exit 1
  echo "order_nodes: shakedown job, continuing with a slower trainer" >&2
fi
echo "${bad[@]}" "${good[@]}"
