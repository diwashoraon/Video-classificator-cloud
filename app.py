import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from gtts import gTTS
from transformers import AutoFeatureExtractor, VideoMAEForVideoClassification

# Flask App
app = Flask(__name__)

# Load Pretrained Model (Hugging Face example)
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)

# Home Page
@app.route("/")
def home():
    return """
    <h2>🎬 Cloud Video Classifier</h2>
    <form action="/classify" method="post" enctype="multipart/form-data">
        <input type="file" name="video" required>
        <input type="submit" value="Classify Video">
    </form>
    """

# Video Classification Endpoint
@app.route("/classify", methods=["POST"])
def classify():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"})
    
    video = request.files['video']
    video_path = os.path.join("temp_video.mp4")
    video.save(video_path)

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()

    # Prepare model input
    inputs = feature_extractor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Voice feedback
    tts = gTTS(text=f"Video classified as category {predicted_class}", lang="en")
    tts.save("result_audio.mp3")

    return jsonify({"predicted_class": predicted_class, "audio_file": "result_audio.mp3"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
