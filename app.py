from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from heartDiseaseClassification.pipeline.prediction import PredictionPipeline
import numpy as np
from aif360.datasets import BinaryLabelDataset

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.data = []
        self.classifier = PredictionPipeline(self.data)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


def preprocess(data):
    age_threshold = 56
    target_variable = "target"
    data["age_binary"] = np.where(data["age"] > age_threshold, 1, 0)
    data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=data,
        label_names=[target_variable],
        protected_attribute_names=["sex", "age_binary"],
    )
    return data


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.json['data']
        data = preprocess(data)
        c1App.data = data
        result = c1App.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify("Error: " + str(e))


if __name__ == "__main__":
    c1App = ClientApp()
    app.run(host='0.0.0.0', port=8000)
