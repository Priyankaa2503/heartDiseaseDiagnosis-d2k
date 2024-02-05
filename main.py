from heartDiseaseClassification import logger
from heartDiseaseClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from heartDiseaseClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from heartDiseaseClassification.pipeline.stage_03_generate_report import GeneratingReportTrainingPipeline

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

STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f"********************************************")
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(
        f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Generating Report"
try:
    logger.info(f"********************************************")
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    generate_report = GeneratingReportTrainingPipeline()
    generate_report.main()
    logger.info(
        f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e
