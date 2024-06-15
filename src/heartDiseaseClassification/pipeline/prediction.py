import pickle
import os


class PredictionPipeline:
    def __init__(self, data):
        self.data = data

    def predict(self):
        model = pickle.load(
            open(os.path.join("artifacts", "prepare_models", "voting_clf.pkl"), "rb"))
        input = self.data
        result = model.predict([input])
        return result


if __name__ == "__main__":
    sample_data = [
        44.0,
        1.0,
        2.0,
        130.0,
        233.0,
        0.0,
        1.0,
        179.0,
        1.0,
        0.4,
        2.0,
        0.0,
        2.0,
        0.0,
    ]
    obj = PredictionPipeline(sample_data)
    sample_pred = obj.predict()

    print("\nSample Predictions:")
    for i, (sample, pred) in enumerate(zip([sample_data], sample_pred)):
        print(f"Sample {i + 1}:")
        print(f"  Features: {sample}")
        # if 1 then print healthy else print not healthy
        print(f"  Prediction: {'Healthy' if pred == 1 else 'Not Healthy'}")
        # print(f"  Prediction: {pred}")
        print()
