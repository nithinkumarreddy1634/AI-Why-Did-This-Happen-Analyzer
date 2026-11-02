"""Application Settings and Configuration Management.

Provides centralized configuration for directories, runtime parameters,
file size constraints, theme options, and session initialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List, Dict, Any

from config.constants import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    APP_DESCRIPTION,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    THEME_COLORS,
)

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"


@dataclass
class ThemeConfig:
    """Theme configuration settings."""
    theme_name: str = "Enterprise Dark"
    colors: Dict[str, str] = field(default_factory=lambda: THEME_COLORS.copy())
    font_family: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    border_radius: str = "8px"
    card_shadow: str = "0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)"


@dataclass
class AppSettings:
    """Centralized application settings."""
    app_name: str = APP_NAME
    app_version: str = APP_VERSION
    tagline: str = APP_TAGLINE
    description: str = APP_DESCRIPTION
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    uploads_dir: Path = UPLOADS_DIR
    processed_dir: Path = PROCESSED_DIR
    samples_dir: Path = SAMPLES_DIR
    
    # Upload & Storage constraints
    supported_extensions: List[str] = field(default_factory=lambda: SUPPORTED_EXTENSIONS.copy())
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    
    # Theme configuration
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    
    # Runtime toggles
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() in ("true", "1", "yes"))
    enable_cache: bool = True
    max_rows_display: int = 1000

    def ensure_directories(self) -> None:
        """Create necessary project data storage directories if missing."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)


# Global singleton instance
_settings_instance: AppSettings | None = None


def get_settings() -> AppSettings:
    """Returns the singleton instance of AppSettings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
        _settings_instance.ensure_directories()
    return _settings_instance
