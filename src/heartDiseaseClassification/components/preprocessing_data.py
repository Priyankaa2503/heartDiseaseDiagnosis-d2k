import pandas as pd
from sklearn.model_selection import train_test_split
from heartDiseaseClassification.entity.config_entity import PreprocessingDataConfig
from pathlib import Path
from heartDiseaseClassification import logger
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aif360.metrics import BinaryLabelDatasetMetric
import numpy as np


class PreprocessingData:
    def __init__(self, config: PreprocessingDataConfig):
        self.config = config

    def preprocess_data(self):
        print(f"Path : {self.config.data_path}")
        data = pd.read_csv(self.config.data_path)
        print(data.head(3))

        age_threshold = data["age"].median()
        unprivileged_groups = [
            {"sex": 0, "age_binary": 0},
            {"sex": 0, "age_binary": 1},
            {"sex": 1, "age_binary": 0},
        ]
        privileged_groups = [{"sex": 1, "age_binary": 1}]
        target_variable = "target"

        data["age_binary"] = np.where(data["age"] > age_threshold, 1, 0)

        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=data,
            label_names=[target_variable],
            protected_attribute_names=["sex", "age_binary"],
        )

        metric_orig = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        print("Original dataset metrics:")
        print(f"Disparate Impact: {metric_orig.disparate_impact()}")
        print(f"Mean Difference: {metric_orig.mean_difference()}")

        RW = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        dataset_transf = RW.fit_transform(dataset)

        metric_transf = BinaryLabelDatasetMetric(
            dataset_transf,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        print("Transformed dataset metrics:")
        print(f"Disparate Impact: {metric_transf.disparate_impact()}")
        print(f"Mean Difference: {metric_transf.mean_difference()}")

        X_train, X_test, y_train, y_test = train_test_split(
            dataset_transf.features,
            dataset_transf.labels.ravel(),
            test_size=0.2,
            random_state=0,
        )

        X_train = pd.DataFrame(X_train, columns=dataset_transf.feature_names)
        X_test = pd.DataFrame(X_test, columns=dataset_transf.feature_names)
        y_train = pd.DataFrame(y_train, columns=[target_variable])
        y_test = pd.DataFrame(y_test, columns=[target_variable])

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
