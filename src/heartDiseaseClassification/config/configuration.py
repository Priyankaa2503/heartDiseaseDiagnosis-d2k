from heartDiseaseClassification.constants import *
from heartDiseaseClassification.utils.common import read_yaml, create_directories
from heartDiseaseClassification.entity.config_entity import (
    DataIngestionConfig, PreprocessingDataConfig, PrepareModelConfig, GenerateReportConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_preprocessing_data_config(self) -> PreprocessingDataConfig:
        config = self.config.preprocessing_data
        create_directories([config.root_dir])
        preprocessing_data_config = PreprocessingDataConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            result_data_path=Path(config.result_data_path),
        )
        return preprocessing_data_config

    def get_prepare_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        create_directories([config.root_dir])
        prepare_model_config = PrepareModelConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path
        )
        return prepare_model_config

    def get_generate_report_config(self) -> GenerateReportConfig:
        config = self.config.generate_report
        create_directories([config.root_dir])
        generate_report_config = GenerateReportConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            report_path=config.report_path
        )
        return generate_report_config
