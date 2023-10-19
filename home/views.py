# app_name/views.py
import joblib
import pandas as pd
from django.shortcuts import render

from datetime import datetime

# Load the emotion detection model
emotion_model = joblib.load(open("emotion.pkl", "rb"))

def home(request):
    if request.method == 'POST':
        raw_text = request.POST.get('raw_text', '')
        prediction = predict_emotions(emotion_model, raw_text)
        return render(request, 'home.html', {'raw_text': raw_text, 'prediction': prediction})
    return render(request, 'home.html')

def predict_emotions(model, text):
    results = model.predict([text])
    return results[0]
