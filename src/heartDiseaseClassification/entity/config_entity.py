from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PreprocessingDataConfig:
    root_dir: Path
    data_path: Path
    result_data_path: Path


@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    data_path: Path
    model_path: Path

@dataclass(frozen=True)
class GenerateReportConfig:
    root_dir: Path
    model_path: Path
    report_path: Path
