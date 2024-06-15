import sys

sys.path.insert(0, "./src")

from heartDiseaseClassification import logger
from heartDiseaseClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from heartDiseaseClassification.pipeline.stage_02_data_preprocessing import PreprocesingDataPipeline
from heartDiseaseClassification.pipeline.stage_03_prepare_model import PrepareModelsTrainingPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    data_ingesetion = DataIngestionTrainingPipeline()
    data_ingesetion.main()
    logger.info(
        f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Preprocessing Data"
try:
    logger.info(f"********************************************")
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    preprocessing_data = PreprocesingDataPipeline()
    preprocessing_data.main()
    logger.info(
        f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Preparing Models"
try:
    logger.info(f"********************************************")
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    generate_report = PrepareModelsTrainingPipeline()
    generate_report.main()
    logger.info(
        f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e
