import os
import glob
import numpy as np
import librosa
import kagglehub
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("Dataset download ho gaya is folder mein:", path)

#    IN RAVDESS  DATA  WE ONY  FIND WAV FILES  SO  WE  WILL  ONLY  LOOK  FOR  WAV FILES
file_list = glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)


def extract_fixed_features(file_path):
    sr = 22050
    audio_data, _ = librosa.load(file_path, sr=sr)

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

for file_path in file_list:
    filename = os.path.basename(file_path)
    parts = filename.split('-')
    if len(parts) >= 3 and parts[0] == '03' and parts[1] == '01':
        emotion_code = parts[2]
        emotion_label = emotions_map.get(emotion_code)
        if emotion_label:
            try:
                ml_feat, dl_feat = extract_fixed_features(file_path)
                X_ml.append(ml_feat)
                X_dl.append(dl_feat)
                y.append(emotion_label)
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

emotion_counts = Counter(y)
for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion}: {count} files")
print(f"\nTotal files: {sum(emotion_counts.values())}")

X_ml_array = np.array(X_ml)
y_array = np.array(y)

X_train_ml, X_test_ml, y_train, y_test = train_test_split(
    X_ml_array,
    y_array,
    test_size=0.2,
    random_state=42,
    stratify=y_array  # Balancing the classes in train and test sets
)

print(f"Train data shape: {X_train_ml.shape}")
print(f"Test data shape: {X_test_ml.shape}")
print(f"Train labels shape: {y_train.shape}")

smote = SMOTE(random_state=42)
X_train_ml_sample, y_train_sample = smote.fit_resample(X_train_ml, y_train)

clf = RandomForestClassifier(max_depth=15, random_state=0)
clf.fit(X_train_ml_sample, y_train_sample)
y_pred = clf.predict(X_test_ml)

accuracy_score(y_test, y_pred, normalize=True)

print(f"Train size: {X_train_ml_sample.shape}")
print(f"Test size: {X_test_ml.shape}")
print(f"Train labels: {Counter(y_train_sample)}")
print(f"Test labels: {Counter(y_test)}")

print(classification_report(y_test, y_pred))


def extract_ml_batch(x):
    return np.array([extract_fixed_features(p)[0] for p in x])


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
print("accuracy", pipeline.score(X_test_paths, y_test))

scores = cross_val_score(pipeline, X_train_paths, y_train, cv=5)
print(f"CV SCORES{scores}")
print(f"MEAN ACCURACY{scores.mean():.2f}")
print(f"STD DEVIATION{scores.std():.2f}")

joblib.dump(pipeline, 'ser_pipeline.pkl')
