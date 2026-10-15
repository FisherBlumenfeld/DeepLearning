from pathlib import Path
from importlib.resources import files
from typing import Tuple

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _find_config_toml() -> Path:
    """Resolve config.toml location via importlib.resources or relative path fallback."""
    if __package__:
        try:
            return Path(str(files(__package__).joinpath("config.toml")))
        except Exception:
            pass
    return Path(__file__).parent / "config.toml"


class DataSettings(BaseModel):
    """Settings for data generation."""

    num_features: int = Field(
        default=1, gt=0, description="Number of feature dimensions"
    )
    num_samples: int = Field(
        default=50, gt=0, description="Number of synthetic data points"
    )
    sigma_noise: float = Field(
        default=0.5, ge=0.0, description="Standard deviation of Gaussian noise"
    )


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = Field(default=16, gt=0, description="Mini-batch size for SGD")
    num_iters: int = Field(
        default=300, gt=0, description="Number of optimization iterations"
    )
    learning_rate: float = Field(
        default=0.1, gt=0.0, description="Learning rate for Adam optimizer"
    )


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (5, 3)
    dpi: int = Field(default=200, gt=0, description="Dots per inch for saved figures")
    output_dir: Path = Path("artifacts")


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()
    plotting: PlottingSettings = PlottingSettings()

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
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
