from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from heartDiseaseClassification.pipeline.prediction import PredictionPipeline
