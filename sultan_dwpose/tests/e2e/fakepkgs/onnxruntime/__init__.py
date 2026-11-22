"""Fake onnxruntime(-gpu) for the end-to-end notebook dry run."""
__version__ = "fake-gpu"


def get_available_providers():
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def preload_dlls():
    return None
