import os
from heartDiseaseClassification import logger
from heartDiseaseClassification.entity.config_entity import PrepareModelConfig
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import pickle


class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def prepare_models(self):
        data_path = Path(self.config.data_path)
        X_train = pd.read_csv(f"{data_path}/X_train.csv")
        X_test = pd.read_csv(f"{data_path}/X_test.csv")
        y_train = pd.read_csv(f"{data_path}/y_train.csv")
        y_test = pd.read_csv(f"{data_path}/y_test.csv")

        dt = DecisionTreeClassifier(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        gb = GradientBoostingClassifier(random_state=42)

        models = [dt, rf, gb]

        for model in models:
            logger.info(f"Training model: {model}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Accuracy of {model} is {accuracy}")
            logger.info(
                f"Classification report of {model} is {classification_report(y_test, y_pred)}")

        voting_clf = VotingClassifier(
            estimators=[('dt', dt), ('rf', rf), ('gb', gb)], voting='hard'
        )

        logger.info(
            f"Training model: {voting_clf} using VotingClassifier and hard voting")
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy of {voting_clf} is {accuracy}")
        logger.info(
            f"Classification report of {voting_clf} is {classification_report(y_test, y_pred)}")
        voting_clf_file_path = Path(self.config.model_path)
        with open(f"{voting_clf_file_path}/voting_clf.pkl", "wb") as f:
            pickle.dump(voting_clf, f)
