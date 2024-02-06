import joblib
import os


class PredictionPipeline:
    def __init__(self, data):
        self.data = data
        self.model

    def predict(self):
        model = joblib.load(os.path.join(
            'artifacts', 'prepare_base_model', 'base_model_updated.joblib'))
        input = self.data
        prediction = model.predict(input)
        print("PREDICTION: ")
        if prediction == 1:
            print("The patient has heart disease")
        else:
            print("The patient does not have heart disease")
