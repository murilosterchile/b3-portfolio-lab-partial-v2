from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_env: str = "development"
    app_name: str = "B3 Portfolio Lab"
    database_url: str = "postgresql+psycopg://portfolio:portfolio_dev_password@localhost:5432/portfolio"
    secret_key: str = "dev-only-change-me"
    cors_origins: str = "http://localhost:3000"
    data_dir: str = "/workspace/data"
    model_dir: str = "/workspace/models"
    qkp_solver: str = "auto"
    demo_mode: bool = True
    valkey_url: str = "redis://localhost:6379/0"

    @property
    def cors_origin_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

    @property
    def secure_cookies(self) -> bool:
        return self.app_env.lower() not in {"development", "test", "local"}

    @field_validator("secret_key")
    @classmethod
    def validate_secret(cls, value: str) -> str:
        if len(value) < 16:
            raise ValueError("SECRET_KEY must have at least 16 characters")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
