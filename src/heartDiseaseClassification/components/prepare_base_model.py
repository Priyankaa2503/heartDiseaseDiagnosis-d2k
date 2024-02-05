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
from heartDiseaseClassification.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        logistic_regression = LogisticRegression(
            random_state=self.config.params_random_state)
        random_forest = RandomForestClassifier(
            random_state=self.config.params_random_state)
        decision_tree = DecisionTreeClassifier(
            random_state=self.config.params_random_state)
        svc = SVC(probability=self.config.params_probability,
                  random_state=self.config.params_random_state)

        self.model = VotingClassifier(estimators=[('lr', logistic_regression), (
            'rf', random_forest), ('dt', decision_tree), ('svc', svc)], voting='soft')

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, X_train_scaled, y_train):
        full_model = model.fit(X_train_scaled, y_train)
        # y_pred = full_model.predict(X_train_scaled)
        # print("Classification Report:\n", classification_report(y_train, y_pred))
        return full_model

    def update_base_model(self):
        filePath = f"{self.config.root_dir.parent}/data_ingestion/my_data.csv"
        data = pd.read_csv(filePath)
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(
            imputer.fit_transform(data), columns=data.columns)
        X = df_imputed.drop('target', axis=1)
        y = df_imputed['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        y_test = pd.DataFrame(y_test, columns=['target'])
        y_train = pd.DataFrame(y_train, columns=['target'])
        X_train_scaled.to_csv(
            f"{self.config.root_dir.parent}/data_ingestion/X_train_scaled.csv", index=False)
        X_test_scaled.to_csv(
            f"{self.config.root_dir.parent}/data_ingestion/X_test_scaled.csv", index=False)
        y_train.to_csv(
            f"{self.config.root_dir.parent}/data_ingestion/y_train.csv", index=False)
        y_test.to_csv(
            f"{self.config.root_dir.parent}/data_ingestion/y_test.csv", index=False)
        self.full_model = self._prepare_full_model(
            model=self.model,
            X_train_scaled=X_train_scaled,
            y_train=y_train
        )
        self.save_model(path=self.config.updated_base_model_path,
                        model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            joblib.dump(model, path)
        except Exception as e:
            raise Exception(f"Error in saving model: {e}")
