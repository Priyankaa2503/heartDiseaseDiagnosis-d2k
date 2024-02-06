from heartDiseaseClassification.config.configuration import ConfigurationManager
from heartDiseaseClassification.components.prepare_model import PrepareModel
from heartDiseaseClassification import logger

STAGE_NAME = "Preparing Models"


class PrepareModelsTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.get_prepare_model_config()
        prepare_model_config = PrepareModel(config=prepare_model_config)
        prepare_model_config.prepare_models()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        obj = PrepareModelsTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>> Stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
