"""Optional lazy local cross-encoder reranker."""
from functools import lru_cache
import math
from threading import Lock

from app.infrastructure.config import Settings


class LocalCrossEncoderReranker:
    provider = "cross_encoder"

    def __init__(self, model: str, revision: str, cache_dir: str, local_files_only: bool,
                 batch_size: int, cpu_threads: int):
        self.model = model
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.cpu_threads = cpu_threads
        self._model = None
        self._lock = Lock()

    def _load(self):
        import torch
        from sentence_transformers import CrossEncoder
        torch.set_num_threads(self.cpu_threads)
        return CrossEncoder(
            self.model, revision=self.revision, device="cpu", cache_folder=self.cache_dir,
            local_files_only=self.local_files_only, trust_remote_code=False, token=False,
            model_kwargs={"use_safetensors": True}, max_length=512,
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            if self._model is None:
                self._model = self._load()
            raw = self._model.predict(
                [(query, passage) for passage in passages], batch_size=self.batch_size,
                show_progress_bar=False, convert_to_numpy=True,
            )
        values = [float(value) for value in raw.reshape(-1).tolist()]
        if len(values) != len(passages) or any(not math.isfinite(value) for value in values):
            raise ValueError("Cross-encoder returned invalid scores")
        return values


@lru_cache(maxsize=4)
def _cross_encoder_provider(model, revision, cache_dir, local_files_only, batch_size, cpu_threads):
    return LocalCrossEncoderReranker(model, revision, cache_dir, local_files_only, batch_size, cpu_threads)


def create_reranker_provider(settings: Settings):
    return _cross_encoder_provider(
        settings.reranker_model, settings.reranker_revision, str(settings.reranker_cache_dir),
        settings.reranker_local_files_only, settings.reranker_batch_size, settings.reranker_cpu_threads,
    )
