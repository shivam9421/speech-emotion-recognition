import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import soundfile as sf
import tempfile

def extract_fixed_features(file_path,duration=3):
    sr=22050
    max_len=int (duration*sr)
    audio_data,sampling_rate=librosa.load(file_path,sr=sr)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    ml_features = np.mean(mfccs.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    dl_features = librosa.power_to_db(mel, ref=np.max)

    return ml_features, dl_features

def extract_ml_batch(x):
    return np.array([extract_fixed_features(p)[0] for p in x])
#----Model Load-----
pipeline=joblib.load('ser_pipeline.pkl')
#----UI-------------
st.title('🎙️ Speech Emotion Recognition')
st.write('Audio upload karo - emotion detect ho jaayega!')
uploaded_file=st.file_uploader("WAV file upload karo",type=['wav'])
st.audio(uploaded_file)
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.wav') as f:
        f.write(uploaded_file.read())
        f_path=f.name
    #audio load
    audio_data,sampling_rate=librosa.load(f_path,sr=22050)
    #-----waveform-------
    st.subheader('waveform')
    fig,ax=plt.subplots(figsize=(10,3))
    librosa.display.waveshow(audio_data,ax=ax,sr=22050)
    ax.set_facecolor('#f0f0f0')
    ax.set_title("Audio Waveform")
    st.pyplot(fig)
    # ── Mel Spectrogram ──
    # st.subheader("🎨 Mel Spectrogram")
    # mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    # mel_db = librosa.power_to_db(mel, ref=np.max)
    # fig2, ax2 = plt.subplots(figsize=(10, 4))
    # img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
    #                                 cmap='viridis', ax=ax2)
    # fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    # st.pyplot(fig2)




    #------predictions
    st.subheader(" Predicted Emotion")
    prediction=pipeline.predict([f_path])
    probablity=pipeline.predict_proba([f_path])
    classes = pipeline.classes_
    emotion_emoji = {
        'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
        'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
    }
    st.markdown(f'predicted emotion :{emotion_emoji[prediction[0]]} {prediction[0].upper()}')
    #------confidence chart
    st.subheader(" Predicted Emotion Probability")
    fig3,ax3=plt.subplots(figsize=(10,4))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    ax3.bar(classes, probablity[0], color=colors)
    ax3.set_ylabel("Confidence")
    ax3.set_title("Emotion Probabilities")
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)






