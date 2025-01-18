import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model and label encoder
model = tf.keras.models.load_model('fake_or_real_audio_lstm_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc_mean, axis=0)

def predict(audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Streamlit UI
st.title('DeepFake Audio Detection')
st.write("Upload an audio file to determine if it's real or fake")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    result = predict(uploaded_file)
    st.write(f'The audio is predicted as: **{result}**')
