"""Central configuration. All settings come from environment variables / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    database_path: Path = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/analyst.db")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    reports_dir: Path = PROJECT_ROOT / "reports"

    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    alert_recipient: str = os.getenv("ALERT_RECIPIENT", "")


settings = Settings()