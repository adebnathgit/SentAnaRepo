# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentiment_analysis_ml import predict_sentiment
# Load the trained model
model_path = 'model_pickle'
with open(model_path, 'rb') as file:
    model_pickle_rest = pickle.load(file)
app = Flask(__name__)

test_sentence="We are welcoming the new era of technology"

predict_sentiment(test_sentence,model_pickle_rest)
