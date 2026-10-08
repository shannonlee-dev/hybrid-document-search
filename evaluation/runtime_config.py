"""Explicit benchmark runtime settings; call before model or query execution."""

import os

DEFAULT_THREADS = 2
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
THREAD_COUNT_KEYS = (
    "torch_num_threads",
    "torch_num_interop_threads",
    "faiss_omp_max_threads",
)
CUDA_DEVICE = "cuda:0"
_LIMITER = None


def configure_runtime(threads: int, *, cuda: bool = False):
    global _LIMITER
    for name in THREAD_ENV:
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import faiss
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != threads:
        torch.set_num_interop_threads(threads)
    faiss.omp_set_num_threads(threads)
    _LIMITER = threadpool_limits(limits=threads)
    result = {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "faiss_omp_max_threads": faiss.omp_get_max_threads(),
        "threadpools": threadpool_info(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if any(result[key] != threads for key in THREAD_COUNT_KEYS) or any(
        pool["num_threads"] != threads for pool in result["threadpools"]
    ):
        raise RuntimeError("effective thread count differs from requested value")
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda:0 is required; CPU fallback is forbidden")
        torch.zeros(1, device=CUDA_DEVICE).sum().item()
        result.update(
            gpu=torch.cuda.get_device_name(0),
            cuda=torch.version.cuda,
            device=CUDA_DEVICE,
        )
    return result
