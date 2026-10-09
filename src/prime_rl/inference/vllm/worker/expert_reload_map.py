"""Avoid per-weight GPU scalar reads while inference is paused for reload."""

from contextlib import contextmanager


@contextmanager
def snapshot_expert_maps(model):
    """Snapshot current placement once, without changing device routing buffers.

    The caller must pause inference and expert rebalancing for the whole reload.
    Maps are refreshed on every invocation and restored even when loading fails.
    """
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    missing = object()
    saved = []
    seen = set()
    try:
        for module in model.modules():
            manager = getattr(module, "expert_map_manager", None)
            if not isinstance(manager, ExpertMapManager) or id(manager) in seen:
                continue
            seen.add(id(manager))
            # Preserve custom implementations and the no-EP identity mapping.
            if (
                "map_global_to_local" in manager.__dict__
                or type(manager).map_global_to_local is not ExpertMapManager.map_global_to_local
                or manager._expert_map is None
            ):
                continue
            placement = tuple(manager._expert_map.detach().cpu().tolist())
            saved.append((manager, manager.__dict__.get("map_global_to_local", missing)))
            manager.map_global_to_local = placement.__getitem__
        yield
    finally:
        for manager, previous in reversed(saved):
            if previous is missing:
                del manager.map_global_to_local
            else:
                manager.map_global_to_local = previous
