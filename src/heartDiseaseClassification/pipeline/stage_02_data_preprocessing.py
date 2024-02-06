from heartDiseaseClassification.config.configuration import ConfigurationManager
from heartDiseaseClassification.components.preprocessing_data import PreprocessingData
from heartDiseaseClassification import logger

STAGE_NAME = "Preprocessing Data"


class PreprocesingDataPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessing_data_config = config.get_preprocessing_data_config()
        preprocess_data = PreprocessingData(
            config=preprocessing_data_config)
        preprocess_data.preprocess_data()


if __name__ == "__main__":
    try:
        logger.info(f"********************************************")
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        obj = PreprocesingDataPipeline()
        obj.main()
        logger.info(
            f">>>>> Stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
