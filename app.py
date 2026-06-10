import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import soundfile as sf
import tempfile
# ── Page Config ──
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)
# ── Custom CSS ──
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .emotion-text {
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .confidence-text {
        color: #4ECDC4;
        font-size: 1.2rem;
    }
    .section-header {
        color: #4ECDC4;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        border-left: 4px solid #4ECDC4;
        padding-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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
#----Emotion config---
emotion_emoji = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}
emotion_color = {
    'neutral': '#95A5A6', 'calm': '#3498DB', 'happy': '#F1C40F',
    'sad': '#2980B9', 'angry': '#E74C3C', 'fearful': '#9B59B6',
    'disgust': '#27AE60', 'surprised': '#E67E22'
}
#----Model Load-----
pipeline=joblib.load('ser_pipeline.pkl')
#--------header---
st.markdown('<p class="title">🎙️ Speech Emotion Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a WAV file and let AI detect the emotion!</p>', unsafe_allow_html=True)
#----Upload-------------
uploaded_file=st.file_uploader("",type=['wav'],label_visibility="collapsed")
# Upload ke UPAR yeh add karo
if uploaded_file is  None:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e1e2e, #2a2a3e); 
                    border-radius: 20px; padding: 30px; text-align: center;
                    border: 1px solid #333; margin-bottom: 2rem;">
            <h2 style="color: #4ECDC4;">How it works?</h2>
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div style="color: white;">
                    <div style="font-size: 2rem;">📤</div>
                    <p>Upload WAV file</p>
                </div>
                <div style="color: #888; font-size: 2rem;">→</div>
                <div style="color: white;">
                    <div style="font-size: 2rem;">🔍</div>
                    <p>AI analyzes audio</p>
                </div>
                <div style="color: #888; font-size: 2rem;">→</div>
                <div style="color: white;">
                    <div style="font-size: 2rem;">🎭</div>
                    <p>Emotion detected!</p>
                </div>
            </div>
            <p style="color: #888; margin-top: 20px;">
                Detects 8 emotions: 😐 Neutral • 😌 Calm • 😊 Happy • 😢 Sad • 
                😠 Angry • 😨 Fearful • 🤢 Disgust • 😲 Surprised
            </p>
        </div>
    """, unsafe_allow_html=True)
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.wav') as f:
        f.write(uploaded_file.read())
        f_path=f.name
    #audio load
    audio_data,sampling_rate=librosa.load(f_path,sr=22050)
    #-----audio player---
    st.markdown('<p class="section-header">🔊 Audio Player</p>', unsafe_allow_html=True)
    st.audio(f_path)
    #------predictions
    prediction=pipeline.predict([f_path])
    probablity=pipeline.predict_proba([f_path])
    classes = pipeline.classes_
    emotion_emoji = {
        'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
        'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
    }
    # ── Emotion Card ──
    st.markdown('<p class="section-header">🎭 Detected Emotion</p>', unsafe_allow_html=True)
    emotion = prediction[0]
    color = emotion_color.get(emotion, '#4ECDC4')
    emoji = emotion_emoji.get(emotion, '🎵')
    confidence = round(max(probablity[0]) * 100, 1)

    st.markdown(f"""
        <div class="emotion-card">
            <div style="font-size:4rem">{emoji}</div>
            <div class="emotion-text" style="color:{color}">{emotion.upper()}</div>
            <div class="confidence-text">Confidence: {confidence}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    #-----waveform-------
    col1,col2=st.columns(2)
    with col1:
        st.markdown('<p class="section-header">📊 Waveform</p>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,3))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        librosa.display.waveshow(audio_data,ax=ax,sr=22050,color='#4ECDC4')
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Amplitude", color='white')
        ax.set_title("Audio Waveform")
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig)

    # ── Mel Spectrogram ──
    with col2:
        st.markdown('<p class="section-header">🎨 Mel Spectrogram</p>', unsafe_allow_html=True)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=22050, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor('#1e1e2e')
        ax2.set_facecolor('#1e1e2e')
        img = librosa.display.specshow(mel_db, sr=22050, x_axis='time', y_axis='mel',cmap='magma', ax=ax2)
        ax2.tick_params(colors='white')
        ax2.set_xlabel("Time (s)", color='white')
        ax2.set_ylabel("Hz", color='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)
    #------confidence chart
    st.markdown('<p class="section-header">📈 Emotion Probabilities</p>', unsafe_allow_html=True)
    colors = [emotion_color.get(c, '#4ECDC4') for c in classes]
    fig3,ax3=plt.subplots(figsize=(12,4))
    fig3.patch.set_facecolor('#1e1e2e')
    ax3.set_facecolor('#1e1e2e')
    ax3.bar(classes, probablity[0], color=colors)
    bars = ax3.bar(classes, probablity[0], color=colors, edgecolor='#333', linewidth=0.5)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Confidence", color='white')
    ax3.set_ylabel("Confidence")
    ax3.tick_params(colors='white')
    for bar, prob in zip(bars, probablity[0]):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{prob:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    ax3.set_title("Emotion Probabilities")
    st.pyplot(fig3)






