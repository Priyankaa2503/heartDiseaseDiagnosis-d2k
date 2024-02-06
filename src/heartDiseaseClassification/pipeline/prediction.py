import pickle
import os


class PredictionPipeline:
    def __init__(self, data):
        self.data = data

    def predict(self):
        model = pickle.load(
            open(os.path.join("models", "heart_disease_model.pkl"), "rb"))
        input = self.data
        result = model.predict(input)

        if result[0] == 1:
           prediction = "The patient has heart disease"
           return [{"result": prediction}]
        else:
            prediction = "The patient does not have heart disease"
            return [{"result": prediction}]
