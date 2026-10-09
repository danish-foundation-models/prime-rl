"""CPU-only NeMo tasksets and a minimal tool-capable harness for verifiers v1."""

__all__ = ["NeMoLocalTaskset", "NeMoLocalHarness"]


def __getattr__(name):
    if name == "NeMoLocalTaskset":
        from .taskset import NeMoLocalTaskset

        return NeMoLocalTaskset
    if name == "NeMoLocalHarness":
        from .harness import NeMoLocalHarness

        return NeMoLocalHarness
    raise AttributeError(name)
