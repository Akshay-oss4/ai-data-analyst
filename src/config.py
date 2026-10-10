"""Central configuration. All settings come from environment variables / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # AI
    ai_provider: str = os.getenv("AI_PROVIDER", "ollama").lower()  # ollama | gemini
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Storage and limits
    database_path: Path = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/analyst.db")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    reports_dir: Path = PROJECT_ROOT / "reports"

    # Email alerts (Phase 10)
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    alert_recipient: str = os.getenv("ALERT_RECIPIENT", "")


settings = Settings()