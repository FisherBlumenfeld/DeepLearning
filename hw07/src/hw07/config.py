"""Configuration for the Assignment 7 sparse autoencoder pipeline."""

from importlib.resources import files
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _find_config_toml() -> Path:
    try:
        return Path(str(files(__package__).joinpath("config.toml")))
    except Exception:
        return Path(__file__).parent / "config.toml"


class DataSettings(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0)
    dataset_name: str = "mteb/banking77"
    num_classes: int = Field(default=77, gt=1)
    embedding_batch_size: int = Field(default=128, gt=0)


class TrainingSettings(BaseModel):
    batch_size: int = Field(default=256, gt=0)
    num_epochs: int = Field(default=40, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    l1_coefficient: float = Field(default=1e-3, ge=0)
    checkpoint_dir: Path = Path("artifacts/checkpoints")


class ExperimentSettings(BaseModel):
    latent_dims: list[int] = Field(default_factory=lambda: [384, 768, 1536])
    sample_sizes: list[int] = Field(default_factory=lambda: [512, 2048, 8192])
    top_k_examples: int = Field(default=8, gt=0)
    near_dead_fraction_threshold: float = Field(default=0.001, ge=0, le=1)


class PlottingSettings(BaseModel):
    figsize: Tuple[int, int] = (14, 4)
    dpi: int = Field(default=180, gt=0)
    output_dir: Path = Path("artifacts")


class AppSettings(BaseSettings):
    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = Field(default_factory=DataSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    experiments: ExperimentSettings = Field(default_factory=ExperimentSettings)
    plotting: PlottingSettings = Field(default_factory=PlottingSettings)

    model_config = SettingsConfigDict(
        toml_file=_find_config_toml(),
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    return AppSettings()
