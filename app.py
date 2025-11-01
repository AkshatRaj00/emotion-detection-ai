import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="🎭 Emotion Detection AI",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .happy { background-color: #FFD700; color: black; }
    .sad { background-color: #4169E1; color: white; }
    .angry { background-color: #FF4500; color: white; }
    .fear { background-color: #9932CC; color: white; }
    .disgust { background-color: #228B22; color: white; }
    .surprise { background-color: #FF69B4; color: white; }
    .neutral { background-color: #A9A9A9; color: white; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎭 Emotion Detection AI</div>', unsafe_allow_html=True)
st.write("Apni face image upload kar emotion detect karo! (48x48 grayscale best hai)")

# Emotion mapping
emotion_map = {
    0: ('Angry', '😠', 'angry'),
    1: ('Disgust', '🤢', 'disgust'),
    2: ('Fear', '😨', 'fear'),
    3: ('Happy', '😊', 'happy'),
    4: ('Sad', '😢', 'sad'),
    5: ('Surprise', '😮', 'surprise'),
    6: ('Neutral', '😐', 'neutral')
}

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('emotion_model_fast.h5')
        return model
    except:
        st.error("❌ Model file not found!")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose image:", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file).convert('L')
        image_resized = image.resize((48, 48))
        st.image(image_resized, caption="Processed (48x48)", width=200)
    
    with col2:
        st.subheader("🔮 Prediction")
        
        # Convert to numpy
        img_array = np.array(image_resized) / 255.0
        img_reshaped = img_array.reshape(1, 48, 48, 1)
        
        # Predict
        prediction = model.predict(img_reshaped, verbose=0)
        emotion_id = np.argmax(prediction)
        confidence = prediction[0][emotion_id] * 100
        
        emotion_name, emoji, color_class = emotion_map[emotion_id]
        
        # Display result
        st.markdown(
            f'<div class="emotion-box {color_class}">'
            f'{emoji} {emotion_name}<br>'
            f'Confidence: {confidence:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Show all predictions
    st.subheader("📊 All Emotion Probabilities")
    
    cols = st.columns(7)
    for i, (name, emoji, _) in emotion_map.values():
        with cols[i]:
            prob = prediction[0][i] * 100
            st.metric(name, f"{prob:.1f}%")

# Sidebar info
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
**Emotion Detection AI**
- Built with TensorFlow
- 7 emotions detected
- Powered by CNN
- Deploy: Streamlit Cloud

**Emotions:**
😠 Angry
🤢 Disgust
😨 Fear
😊 Happy
😢 Sad
😮 Surprise
😐 Neutral
""")

st.sidebar.header("📞 Links")
st.sidebar.write("[GitHub](https://github.com/yourname/emotion-detection-ai)")
st.sidebar.write("[LinkedIn](https://linkedin.com/in/yourprofile)")
