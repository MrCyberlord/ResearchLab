import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "level": record.levelname, "logger": record.name,
                   "event": record.getMessage()}
        for key in ("method", "status_code", "duration_ms", "retrieval_ms", "generation_ms",
                    "rerank_ms", "retrieval_strategy", "semantic_candidate_count", "keyword_candidate_count",
                    "retrieved_chunk_count", "context_chunk_count", "prompt_tokens", "completion_tokens", "outcome",
                    "agent_route", "agent_node", "agent_step_count"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("researchlab")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
