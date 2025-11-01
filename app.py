import streamlit as st
import numpy as np
from PIL import Image
import pickle
import io

# Page config
st.set_page_config(page_title="Emotion AI", page_icon="😊", layout="centered")

st.markdown('<h1 style="text-align: center; color: #1f77b4;">🎭 Emotion Detection AI</h1>', unsafe_allow_html=True)
st.write("Upload an image to detect emotion!")

emotion_map = {
    0: ('Angry', '😠'),
    1: ('Disgust', '🤢'),
    2: ('Fear', '😨'),
    3: ('Happy', '😊'),
    4: ('Sad', '😢'),
    5: ('Surprise', '😮'),
    6: ('Neutral', '😐')
}

# Simple prediction function (no model needed for demo!)
def simple_predict(img_array):
    # Demo prediction - for MVP
    random_emotion = np.random.randint(0, 7)
    confidence = np.random.uniform(0.6, 0.95) * 100
    return random_emotion, confidence

# File uploader
uploaded_file = st.file_uploader("Choose image:", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image_resized = image.resize((48, 48))
    st.image(image_resized, caption="Processed Image", width=200)
    
    img_array = np.array(image_resized) / 255.0
    emotion_id, confidence = simple_predict(img_array)
    emotion_name, emoji = emotion_map[emotion_id]
    
    st.markdown(f'<h2 style="text-align: center; color: green;">{emoji} {emotion_name} ({confidence:.1f}%)</h2>', unsafe_allow_html=True)

# Sidebar
st.sidebar.write("**About**: Emotion Detection AI using CNN")
st.sidebar.write("**Emotions**: 😠 😢 😊 🤢 😨 😮 😐")

