import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib
from heartDiseaseClassification.entity.config_entity import PreprocessingDataConfig
from pathlib import Path
from heartDiseaseClassification import logger


class PreprocessingData:
    def __init__(self, config: PreprocessingDataConfig):
        self.config = config

    def preprocess_data(self):
        print(f"Path : {self.config.data_path}")
        data = pd.read_csv(self.config.data_path)
        print(data.head(3))
        X = data.drop("target", axis=1)
        y = data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train.to_csv(
            f"{self.config.result_data_path}/X_train.csv", index=False)
        X_test.to_csv(
            f"{self.config.result_data_path}/X_test.csv", index=False)
        y_train.to_csv(
            f"{self.config.result_data_path}/y_train.csv", index=False)
        y_test.to_csv(
            f"{self.config.result_data_path}/y_test.csv", index=False)

        logger.info(f"\n\nX train shape: {X_train.shape}")
        logger.info(f"\n\nX test shape: {X_test.shape}")
        logger.info(f"\n\ny train shape: {y_train.shape}")
        logger.info(f"\n\ny test shape: {y_test.shape}")
        logger.info(
            f"Preprocessing completed and saved to {self.config.result_data_path}")
