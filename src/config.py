from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(..., alias="CONNECTION_STRING")
    data_dir: Path = Field(default=Path("data/contexts"), alias="DATA_DIR")
    admin_token: str | None = Field(default=None, alias="ADMIN_TOKEN")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")
    user_name: str = Field(default="the user", alias="USER_NAME")  # how prompts refer to you
    compaction_token_threshold: int = Field(default=3000, alias="COMPACTION_TOKEN_THRESHOLD")
    # Public HTTPS address (e.g. a Tailscale Funnel URL); with ADMIN_TOKEN it turns on OAuth for MCP.
    public_url: str | None = Field(default=None, alias="PUBLIC_URL")
    oauth_store: Path = Field(default=Path.home() / ".config/engram/oauth.json", alias="OAUTH_STORE")


# Export .env to os.environ too: provider keys (GEMINI_API_KEY, …) and mem0 read the environment.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

settings = Settings()
