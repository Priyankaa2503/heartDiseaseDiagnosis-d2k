from heartDiseaseClassification import logger
from heartDiseaseClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    data_ingesetion = DataIngestionTrainingPipeline()
    data_ingesetion.main()
    logger.info(
        f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e
