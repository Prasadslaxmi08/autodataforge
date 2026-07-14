"""Configuration system (System Design §5.2).

Layered 12-factor config, precedence high -> low:

    init kwargs  >  environment (VDS_*)  >  .env  >  vds.toml  >  built-in defaults

Config owns *infrastructure and model selection*. Dataset semantics (ontology,
per-class prompts) live in the versioned LabelingPlan instead — the rule is:
if changing a value should trigger re-labeling, it belongs in the plan.

Parsed once into a frozen Settings at startup; invalid config fails here, with a
precise error, never at image 40,000.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_CONFIG_FILE = Path("vds.toml")


class ModelSelection(BaseModel):
    """Capability -> adapter import path (System Design §5.1).

    Values are `"module.path:ClassName"` strings resolved by the ModelRegistry.
    Defaults point at the framework's fake adapter so the app runs with no
    weights installed.
    """

    detector: str = "vds.models.adapters.builtin:BuiltinAdapter"
    segmenter: str = "vds.models.adapters.builtin:BuiltinAdapter"
    embedder: str = "vds.models.adapters.fake:FakeAdapter"
    classifier: str = "vds.models.adapters.fake:FakeAdapter"
    vision_judge: str = "vds.models.adapters.fake:FakeAdapter"
    text_llm: str = "vds.models.adapters.fake:FakeAdapter"


class GpuSettings(BaseModel):
    device: str = "cpu"  # "cuda", "cuda:0", or "cpu"
    vram_budget_mb: int = 8192  # the 8 GB floor (NFR-2)


class RuntimeSettings(BaseModel):
    batch_size: int = 16
    verification_sample_rate: float = 0.25  # stratified sampling default (review §6)
    review_budget_hours: float = 8.0


class StorageSettings(BaseModel):
    database_url: str = "postgresql+psycopg://vds:vds@localhost:5432/vds"
    cas_root: Path = Path("./data/cas")


class ExportSettings(BaseModel):
    default_format: str = "coco"  # coco | yolo | voc


class LLMSettings(BaseModel):
    """Agent-framework provider selection. The provider is chosen entirely here;
    no code outside the provider layer knows which model is used (phase brief).
    Default is the runnable, no-credentials Echo provider."""

    provider: str = "vds.agents.providers.echo:EchoProvider"  # module:ClassName
    model: str = "echo-model"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    base_url: str | None = None
    api_key: str | None = None  # from env VDS_LLM__API_KEY; never committed


class Settings(BaseSettings):
    """The single application configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="VDS_",
        env_nested_delimiter="__",  # VDS_GPU__DEVICE=cuda
        env_file=".env",
        toml_file=_CONFIG_FILE,
        frozen=True,
        extra="forbid",
    )

    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = True

    models: ModelSelection = Field(default_factory=ModelSelection)
    gpu: GpuSettings = Field(default_factory=GpuSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order = precedence, highest first.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Return the process-wide Settings singleton (parsed once)."""
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings
