"""Training runtime diagnostics without device work at module import time."""

from importlib.metadata import PackageNotFoundError, version
import platform
import sys


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def report_training_runtime(device: str) -> dict[str, object]:
    """Validate the requested device and report it before expensive setup.

    Call from a training entry point, never at module import or worker startup.
    CPU runs do not query CUDA devices. CUDA validation queries the driver and
    device properties but does not launch a benchmark or change the device.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unsupported training device: {device!r}")

    import torch

    runtime: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_geometric_version": _package_version("torch-geometric"),
        "numpy_version": _package_version("numpy"),
        "torch_cuda_version": torch.version.cuda,
        "device": device,
        "cuda_checked": device == "cuda",
        "cuda_available": None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    print(f"[runtime] python={runtime['python_executable']}", flush=True)
    print(
        f"[runtime] Python={runtime['python_version']} "
        f"platform={runtime['platform']}",
        flush=True,
    )
    print(
        f"[runtime] torch={runtime['torch_version']} "
        f"PyG={runtime['torch_geometric_version']} "
        f"NumPy={runtime['numpy_version']} "
        f"torch CUDA build={runtime['torch_cuda_version']}",
        flush=True,
    )

    if device == "cuda":
        guidance = (
            f"Requested --device cuda with Python {sys.executable!r}, "
            f"PyTorch {torch.__version__}, and torch.version.cuda="
            f"{torch.version.cuda!r}. Verify that this interpreter uses the "
            "Windows CUDA-enabled PyTorch environment and a compatible NVIDIA "
            "driver. Installing the CUDA Toolkit alone does not enable CUDA "
            "in a CPU-only PyTorch build. Use --device cpu for an explicit CPU "
            "run; no automatic CPU fallback was applied."
        )
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build has no CUDA support. " + guidance)
        try:
            runtime["cuda_available"] = torch.cuda.is_available()
            if not runtime["cuda_available"]:
                raise RuntimeError("torch.cuda.is_available() returned False")
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            runtime.update({
                "cuda_device_index": device_index,
                "cuda_device_name": properties.name,
                "cuda_device_capability": [properties.major, properties.minor],
                "cuda_device_memory_bytes": properties.total_memory,
                "torch_cuda_arch_list": torch.cuda.get_arch_list(),
            })
        except (RuntimeError, OSError, AssertionError) as exc:
            raise RuntimeError(f"CUDA device initialization failed: {exc}. {guidance}") from exc
        print(
            f"[runtime] device=cuda:{runtime['cuda_device_index']} "
            f"GPU={runtime['cuda_device_name']} "
            f"compute capability={properties.major}.{properties.minor}",
            flush=True,
        )
    else:
        print("[runtime] device=cpu (CUDA devices not queried)", flush=True)

    return runtime
