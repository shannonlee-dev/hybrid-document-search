"""Shared constants for retrieval metrics, latency measurement and runtime settings."""

EVALUATION_TOP_K = 10
METRIC_NAMES = ("recall@5", "recall@10", "mrr@10", "ndcg@10")

DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 5
MILLISECONDS_PER_SECOND = 1000

DEFAULT_THREADS = 2
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
THREAD_COUNT_KEYS = (
    "torch_num_threads",
    "torch_num_interop_threads",
    "faiss_omp_max_threads",
)
CUDA_DEVICE = "cuda:0"

CONDITIONS = {
    "device": CUDA_DEVICE,
    "batch_size": 1,
    "dtype": "float32",
    "normalize_embeddings": True,
    "top_k": EVALUATION_TOP_K,
    "warmup": DEFAULT_WARMUP,
    "repeats": DEFAULT_REPEATS,
    "threads": DEFAULT_THREADS,
    "seed": 42,
}
DATASET_SETTINGS = {
    "corpus_size": 10000,
    "train_queries": 100,
    "dev_queries": 50,
    "seed": CONDITIONS["seed"],
}
