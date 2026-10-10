"""Explicit benchmark runtime settings; call before model or query execution."""

import os

from evaluation.constants import CUDA_DEVICE as CUDA_DEVICE
from evaluation.constants import DEFAULT_THREADS as DEFAULT_THREADS
from evaluation.constants import THREAD_COUNT_KEYS as THREAD_COUNT_KEYS
from evaluation.constants import THREAD_ENV as THREAD_ENV

_LIMITER = None


def configure_sparse_runtime(threads: int):
    """Limit Sparse numerical libraries without importing Dense dependencies."""
    global _LIMITER
    for name in THREAD_ENV:
        os.environ[name] = str(threads)
    # 스레드 풀 검사에 포함되도록 수치 연산 라이브러리를 먼저 로드한다.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_info, threadpool_limits

    _LIMITER = threadpool_limits(limits=threads)
    pools = threadpool_info()
    if any(pool["num_threads"] != threads for pool in pools):
        raise RuntimeError("effective thread count differs from requested value")
    return {"threadpools": pools}


def configure_runtime(threads: int, *, cuda: bool = False):
    """Set and verify library thread limits, requiring CUDA when requested.

    Call before model execution; this changes process-wide numerical settings.
    """
    global _LIMITER
    for name in THREAD_ENV:
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import faiss
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    # FP32 비교 실험에서 GPU가 정밀도가 낮은 TF32 연산을 선택하지 않게 한다.
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
        # 장치 감지만 통과한 상태를 배제하고 실제 CUDA 연산 완료까지 확인한다.
        torch.zeros(1, device=CUDA_DEVICE).sum().item()
        result.update(
            gpu=torch.cuda.get_device_name(0),
            cuda=torch.version.cuda,
            device=CUDA_DEVICE,
        )
    return result
