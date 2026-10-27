"""Swappable CPU adapters. Documents and queries share one encoding path."""
from functools import lru_cache
from threading import Lock
import hashlib
import math
import re
import unicodedata

from app.infrastructure.config import Settings
from app.domain.embeddings import LEXICAL_PROFILE, profile_id


class LocalHashEmbeddings:
    model = "local-hash-v1"
    dimension = 384
    revision = "local-hash-v1"
    profile = LEXICAL_PROFILE

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
            for token in tokens or [text]:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dimension
                vector[bucket] += 1.0 if digest[8] & 1 else -1.0
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:  # Hash collisions can cancel every feature.
                vector[0] = 1.0
                norm = 1.0
            result.append([value / norm for value in vector])
        return result


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class MiniLMEmbeddings:
    dimension = 384

    def __init__(self, model: str, revision: str, cache_dir: str, local_files_only: bool,
                 batch_size: int, cpu_threads: int):
        self.model = model
        self.revision = revision
        self.profile = profile_id(model, revision, "nfkc-whitespace-mean-pooling-f32-v1", 256)
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.cpu_threads = cpu_threads
        self._encoder = None
        self._lock = Lock()

    @staticmethod
    def preprocess(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).replace("\x00", "").split())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self.preprocess(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Cannot embed empty text")
        # Single shared model per configuration/process; bound CPU concurrency.
        with self._lock:
            if self._encoder is None:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(self.cpu_threads)
                encoder = SentenceTransformer(
                    self.model, revision=self.revision, device="cpu", cache_folder=self.cache_dir,
                    local_files_only=self.local_files_only, trust_remote_code=False, token=False,
                    model_kwargs={"use_safetensors": True},
                )
                encoder.max_seq_length = 256
                if encoder.get_sentence_embedding_dimension() != self.dimension:
                    raise ValueError("Model output dimension must be 384")
                self._encoder = encoder
            return self._encoder.encode(prepared, batch_size=self.batch_size,
                normalize_embeddings=True, precision="float32", convert_to_numpy=True,
                show_progress_bar=False, prompt="").tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=4)
def _minilm_provider(model, revision, cache_dir, local_files_only, batch_size, cpu_threads):
    return MiniLMEmbeddings(model, revision, cache_dir, local_files_only, batch_size, cpu_threads)


def create_embedding_provider(settings: Settings):
    if settings.embedding_provider == "local":
        return LocalHashEmbeddings()
    return _minilm_provider(settings.embedding_model, settings.embedding_revision,
                            str(settings.embedding_cache_dir), settings.embedding_local_files_only,
                            settings.embedding_batch_size, settings.embedding_cpu_threads)


if __name__ == "__main__":
    provider = create_embedding_provider(Settings())
    provider.embed_query("Warm up the local embedding model.")
    print(f"Ready: {provider.model}@{provider.revision}, {provider.dimension} dimensions; profile {provider.profile}")
