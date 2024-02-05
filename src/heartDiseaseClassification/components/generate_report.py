from pathlib import Path
import joblib
from sklearn.metrics import classification_report
import pandas as pd
from heartDiseaseClassification.entity.config_entity import GenerateReportConfig


class GenerateReport:
    def __init__(self, config: GenerateReportConfig):
        self.config = config

    def generate_report(self):
        full_model = joblib.load(self.config.model_path)
        root_dir = Path(self.config.root_dir)
        X_train_scaled = pd.read_csv(
            f"{root_dir.parent}/data_ingestion/X_train_scaled.csv")
        y_train = pd.read_csv(f"{root_dir.parent}/data_ingestion/y_train.csv")
        y_pred = full_model.predict(X_train_scaled)
        report = classification_report(y_train, y_pred)

        with open(self.config.report_path, "w") as f:
            f.write(report)
