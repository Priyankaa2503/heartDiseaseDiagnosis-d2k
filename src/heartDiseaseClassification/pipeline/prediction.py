import pickle
import os


class PredictionPipeline:
    def __init__(self, data):
        self.data = data
        self.model

    def predict(self):
        model = pickle.load(
            open(os.path.join("models", "heart_disease_model.pkl"), "rb"))
        input = self.data
        prediction = model.predict(input)
        print("PREDICTION: ")
        if prediction == 1:
            print("The patient has heart disease")
        else:
            print("The patient does not have heart disease")
