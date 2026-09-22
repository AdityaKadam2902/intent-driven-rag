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

    def validate_weights(self) -> None:
        total = self.vector_weight + self.text_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"vector_weight + text_weight should sum to 1.0, got {total}"
            )


@lru_cache
def get_engine_config() -> EngineConfig:
    return EngineConfig()
