





import numpy as np
import pandas as pd
import kagglehub
import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("Dataset download ho gaya is folder mein:", path)

#Folder ke andar ghus kar saari .wav files ki list nikalte hain
# RAVDESS mein Actor_01, Actor_02 jaise subfolders hote hain, isliye '**' use kiya hai
file_list= glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)

# 3. Check karte hain ki files mili ya nahi, aur pehli file ko plot karte hain
if len(file_list) > 0:
    sample_file = file_list[0] # Test karne ke liye pehli file utha li
    print(f"Total files mili: {len(file_list)}")
    print(f"Hum is file ko test kar rahe hain: {os.path.basename(sample_file)}")

    # Ab librosa mein actual file ka path pass hoga
    data, sampling_rate = librosa.load(sample_file)

    # 4. Audio wave dekhne ke liye Plot karein
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.title(f'Audio Waveform - {os.path.basename(sample_file)}')
    plt.show()
else:
    print("Oops! Folder mein koi .wav file nahi mili. Ek baar check karo ki download sahi se hua ya nahi.")


# 1. Apne dataset se koi bhi ek audio file ka path yahan dalo
sample_file = "/kaggle/input/datasets/uwrfkaggler/ravdess-emotional-speech-audio/Actor_01/03-01-01-01-01-01-01.wav"
# 2. Audio load kiya
audio_data, sr = librosa.load(sample_file, sr=22050)

# 3. Mel-Spectrogram nikaala aur use Decibels (dB) mein badla
mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

# 4. Matplotlib Graph Plotting Setup
plt.figure(figsize=(12, 8))

# ---- GRAPH 1: WAVEPLOT ----
plt.subplot(2, 1, 1) # Do graphs mein se pehla graph
librosa.display.waveshow(audio_data, sr=sr, color="royalblue")
plt.title("1. Waveplot (Aawaz ki Loudness aur Waves)", fontsize=14, fontweight='bold')
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude (Loudness)")

# ---- GRAPH 2: MEL-SPECTROGRAM ----
plt.subplot(2, 1, 2) # Do graphs mein se doosra graph
# cmap='viridis' se ranges rang-birangi (yellow/purple) dikhengi
img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
plt.title("2. Mel-Spectrogram (Aawaz ka Digital Fingerprint)", fontsize=14, fontweight='bold')
plt.xlabel("Time (Seconds)")
plt.ylabel("Frequency (Hz - Mel Scale)")


# Side mein colorbar dikhane ke liye jo batayega kaunsa rang kitna loud hai
plt.colorbar(img, format="%+2.0f dB")

# Layout saaf karne ke liye
plt.tight_layout()
plt.show()























  
def extract_fixed_features(file_path, duration=3):
    sr = 22050
    max_len = sr * duration
    audio_data, _ = librosa.load(file_path, sr=sr)

    # Audio Length Fix (Padding/Truncating)
    # if len(audio_data) < max_len:
    #     audio_data = np.pad(audio_data, (0, max_len - len(audio_data)), 'constant')
    # else:
    #     audio_data = audio_data[:max_len]

    # ML Feature (MFCC) MFCC (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    ml_features = np.mean(mfccs.T, axis=0)  

    # DL Feature (Spectrogram)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    dl_features = librosa.power_to_db(mel, ref=np.max)

    return ml_features, dl_features









emotions_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
X_ml, X_dl, y = [], [], []
file_paths = []
#print(f"Processing shuru ho rahi hai... Total {len(file_list)} files hain.")
for file_path in file_list:
  filename = os.path.basename(file_path)
  parts = filename.split('-')
  if len(parts) >= 3 and parts[0] == '03' and parts[1] == '01':#len
    emotion_code = parts[2]
    emotion_label = emotions_map.get(emotion_code)
    if emotion_label:
      try:
                # Pichle step ka fixed feature extraction lagao
                ml_feat, dl_feat = extract_fixed_features(file_path)
                X_ml.append(ml_feat)
                X_dl.append(dl_feat)
                y.append(emotion_label)
                file_paths.append(file_path)  
          
      except Exception as e:
                print(f"Error reading {filename}: {e}")






len(y)


from collections import Counter

emotion_counts = Counter(y)
for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion}: {count} files")

print(f"\nTotal files: {sum(emotion_counts.values())}")


import numpy as np
from sklearn.model_selection import train_test_split


X_ml_array = np.array(X_ml)
y_array = np.array(y)


X_train_ml, X_test_ml, y_train, y_test = train_test_split(
    X_ml_array,
    y_array,
    test_size=0.2,
    random_state=42,
    stratify=y_array #Balance barabar rakhne ke liye!

)


print(f"Train data shape: {X_train_ml.shape}")
print(f"Test data shape: {X_test_ml.shape}")
print(f"Train labels shape: {y_train.shape}")





from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)
X_train_ml_sample,y_train_sample=smote.fit_resample(X_train_ml,y_train)






from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(max_depth=15, random_state=0)
clf.fit(X_train_ml_sample, y_train_sample)
y_pred=clf.predict(X_test_ml)

accuracy_score( y_test, y_pred, normalize=True)





print(f"Train size: {X_train_ml_sample.shape}")
print(f"Test size: {X_test_ml.shape}")
print(f"Train labels: {Counter(y_train_sample)}")
print(f"Test labels: {Counter(y_test)}")


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


from sklearn import set_config
set_config(display='diagram')


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
def extract_ml_batch(x):
    return np.array([extract_fixed_features(p)[0] for p in x  ])
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    file_paths, np.array(y),
    test_size=0.2, random_state=42, stratify=np.array(y)
)    

pipeline = Pipeline([
    ('features', FunctionTransformer(extract_ml_batch)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(max_depth=15))
])

pipeline.fit(X_train_paths, y_train)


print("accuracy",pipeline.score(X_test_paths,y_test))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X_train_paths, y_train, cv=5)
print(f"CV SCORES{scores}")
print(f"MEAN ACCURACY{scores.mean():.2f}")



print(f"STD DEVIATION{scores.std():.2f}")


import joblib
joblib.dump(pipeline, 'ser_pipeline.pkl')





# print(f"Train data shape: {len(X_dl[0])}")



# import numpy as np
# X_dl_array=np.array(X_dl)
# X_dl_array.shape
# X_dl_array = X_dl_array[..., np.newaxis]#ye dimension ke saath khelne ke liye 3d converted in 4d


from sklearn.model_selection import train_test_split

# X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
#     X_dl_array,
#     y_array,
#     test_size=0.2,
#     random_state=5,
#     stratify=y_array#Balance barabar rakhne ke liye!

# )

# print(f"Train shape: {X_train_dl.shape}")
# print(f"Test shape: {X_test_dl.shape}")








# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train_dl)
# y_test_encoded = le.transform(y_test_dl)

# print(f"Classes: {le.classes_}")
# print(f"Sample encoding: happy → {le.transform(['happy'])}")



# print(f"Classes: {le.classes_}")`


# import tensorflow as tf
# from tensorflow.keras import layers, models

# model = models.Sequential([
#     # Block 1
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 130, 1)),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
#     layers.Dropout(0.25),

#     # Block 2
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
#     layers.Dropout(0.25),

#     # Block 3
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
#     layers.Dropout(0.25),

#     # Flatten + Dense
#     layers.Flatten(),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(8, activation='softmax')  # 8 emotions
# ])

# model.summary()


# from sklearn.metrics import classification_report,confusion_matrix

# #for dl
# dl_predict=model.predict(X_test_dl).argmax(axis=1)




# print(classification_report(y_test_encoded,dl_predict))
