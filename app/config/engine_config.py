from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="ENGINE_")

    retrieve_top_n: int = 20
    rerank_top_n: int = 5
    vector_weight: float = 0.7
    text_weight: float = 0.3

    high_confidence_threshold: float = 0.6
    medium_confidence_threshold: float = 0.4

    gemini_breaker_failure_threshold: int = 3
    gemini_breaker_recovery_timeout_seconds: float = 30.0

    postgres_breaker_failure_threshold: int = 5
    postgres_breaker_recovery_timeout_seconds: float = 20.0

    ingestion_call_pacing_seconds: float = 4.5

    # --- Caching ---
    # Redis connection - non-secret, safe to default to localhost for
    # local dev; override via ENGINE_REDIS_HOST etc for other envs.
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Layer 0 (exact match) and Layer 1 (entity-guarded semantic) TTLs,
    # in seconds. TTL is the safety-net backstop, NOT the primary
    # invalidation mechanism - see cache/invalidation.py. Version-tied
    # invalidation on document update is primary; this just bounds the
    # worst case if an invalidation event is ever missed.
    cache_ttl_seconds: int = 86400  # 24 hours

    # Layer 1 semantic similarity threshold - a candidate below this
    # cosine similarity is never considered a match, regardless of
    # entity agreement. See the caching architecture doc's Section on
    # threshold tuning: 0.85-0.90 is the recommended balanced range,
    # since the entity guard (not this threshold alone) is what
    # screens out the dangerous false-positive category.
    cache_semantic_similarity_threshold: float = 0.88

    def validate_weights(self) -> None:
        total = self.vector_weight + self.text_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"vector_weight + text_weight should sum to 1.0, got {total}"
            )


@lru_cache
def get_engine_config() -> EngineConfig:
    return EngineConfig()
