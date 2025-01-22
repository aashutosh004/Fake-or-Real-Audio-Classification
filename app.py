import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model and label encoder
model = tf.keras.models.load_model('fake_or_real_audio_lstm_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to extract features from the audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc_mean, axis=0)

# Function to predict if the audio is fake or real
def predict(audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Streamlit UI
st.set_page_config(page_title="DeepFake Audio Detection", layout="centered")

# Header
st.markdown(
    """
    <div style="background-color:#f7f7f7;padding:10px;border-radius:10px;text-align:center">
        <h1 style="color:#333">DeepFake Audio Detection</h1>
        <p style="color:#555">Upload an audio file to determine if it's <strong>Real</strong> or <strong>Fake</strong>. 
        The model uses advanced RNN (LSTM) to detect audio-based misinformation.</p>
    </div>
    """, unsafe_allow_html=True
)

# File uploader
st.markdown("### Step 1: Upload Your Audio File")
uploaded_file = st.file_uploader(
    "Supported formats: WAV, MP3, FLAC", 
    type=["wav", "mp3", "flac"]
)

# Prediction and display
if uploaded_file is not None:
    st.markdown("### Step 2: Audio Playback")
    st.audio(uploaded_file, format='audio/wav')

    st.markdown("### Step 3: Prediction Results")
    with st.spinner('Analyzing the audio...'):
        result = predict(uploaded_file)
    st.success(f"The audio is predicted as: **{result}**")
else:
    st.info("Upload an audio file to get started!")
