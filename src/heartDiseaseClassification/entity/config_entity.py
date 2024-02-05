from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_test_size: float
    params_random_state: int
    params_probability: bool


@dataclass(frozen=True)
class GenerateReportConfig:
    root_dir: Path
    model_path: Path
    report_path: Path
