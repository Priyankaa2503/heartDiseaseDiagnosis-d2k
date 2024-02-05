from heartDiseaseClassification.config.configuration import ConfigurationManager
from heartDiseaseClassification.components.generate_report import GenerateReport
from heartDiseaseClassification import logger

STAGE_NAME = "Generating Report"


class GeneratingReportTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        generate_report_config = config.get_generate_report_config()
        generate_report = GenerateReport(config=generate_report_config)
        generate_report.generate_report()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        obj = GeneratingReportTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>> Stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
