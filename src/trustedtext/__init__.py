__all__ = ["Config", "TrustedText"]


def __getattr__(name):
    # Lazy so that lightweight submodules (e.g. bench.metrics) don't import torch
    if name in __all__:
        from trustedtext import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'trustedtext' has no attribute {name!r}")
