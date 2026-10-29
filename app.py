import os
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tempfile


def extract_fixed_features(file_path):
    sr = 22050
    audio_data, sampling_rate = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    ml_features = np.mean(mfccs.T, axis=0)
    return ml_features


# Must keep this name: the saved pipeline looks it up by name when loading.
def extract_ml_batch(x):
    return np.array([extract_fixed_features(p) for p in x])


# ----Model Load (cached: loads once, not on every rerun)-----
@st.cache_resource
def load_pipeline():
    return joblib.load('ser_pipeline.pkl')


pipeline = load_pipeline()

# ----UI-------------
st.title('🎙️ Speech Emotion Recognition')
st.write('Audio upload karo - emotion detect ho jaayega!')

uploaded_file = st.file_uploader("WAV file upload karo", type=['wav'])
st.audio(uploaded_file)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(uploaded_file.read())
        f_path = f.name

    try:
        # audio load + basic validation
        audio_data, sampling_rate = librosa.load(f_path, sr=22050)
        duration = len(audio_data) / 22050
        if duration < 0.5 or duration > 10 or np.max(np.abs(audio_data)) < 1e-4:
            raise ValueError("Please upload a clear (non-silent) WAV clip between 0.5 and 10 seconds.")

        # -----waveform-------
        st.subheader('waveform')
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(audio_data, ax=ax, sr=22050)
        ax.set_facecolor('#f0f0f0')
        ax.set_title("Audio Waveform")
        st.pyplot(fig)

        # ------predictions (single pass: features are extracted once)
        st.subheader(" Predicted Emotion")
        probablity = pipeline.predict_proba([f_path])
        classes = pipeline.classes_
        prediction = classes[np.argmax(probablity, axis=1)]

        emotion_emoji = {
            'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
            'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
        }
        st.markdown(f'predicted emotion :{emotion_emoji[prediction[0]]} {prediction[0].upper()}')

        # ------confidence chart
        st.subheader(" Predicted Emotion Probability")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        ax3.bar(classes, probablity[0], color=colors)
        ax3.set_ylabel("Confidence")
        ax3.set_title("Emotion Probabilities")
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Could not process this file: {e}")

    finally:
        # always delete the temp file
        os.remove(f_path)
