from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- secrets: required, no fallback values ---
    APP_USERNAME: str
    APP_PASSWORD: str
    SECRET_KEY: str
    WEB_DATABASE_URL: str

    # --- non-sensitive config: safe defaults ---
    SESSION_COOKIE_NAME: str = "meldep_session"
    SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 12  # 12 hours
    UPLOAD_STAGING_DIR: str = "web/data/staging"


@lru_cache
def get_settings() -> Settings:
    return Settings()
