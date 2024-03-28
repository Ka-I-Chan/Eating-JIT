import os
import copy
import json
import numpy as np
import textgrid
import soundfile as sf
import librosa
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SVMSMOTE

def split_data():
    filenames = ['240306_e5', '240306_e6', '240306_e7', '240306_e8', '240306_e9', '240306_e10', '240306_e11',
                 '240306_e12', '240306_e13', '240306_e14', '240306_e15', '240306_e16', '240307_e1', '240307_e2',
                 '240307_e3', '240307_e4', '240307_e5', '240307_e6']
    for filename in filenames:
        tg = textgrid.TextGrid()
        tg.read('./data/' + filename + '.TextGrid')
        data = tg.tiers
        for tier in data:
            for interval in tier:
                if interval.mark != '':
                    print(tier.name, interval.mark, interval.minTime, interval.maxTime)
                    audio, sr = librosa.load('./data/' + filename + '.wav', sr=None)
                    start_time = interval.minTime
                    end_time = interval.maxTime
                    start_frame = int(start_time * sr)
                    end_frame = int(end_time * sr)
                    segment = audio[start_frame:end_frame]
                    sf.write('./data/' + tier.name + '/' + filename + '_' + str(start_time) + '_' + str(end_time) + '.wav', segment, sr)
                
def mean_of_chroma_vector(audio, sr):
    chroma = librosa.feature.chroma_cens(y=audio, sr=sr)
    # print(chroma.mean())
    return chroma.mean()

def root_mean_quare_energy(audio, sr):
    rmse = librosa.feature.rms(y=audio, frame_length=min(2048, len(audio)))
    # print(rmse.mean())
    return rmse.mean()

def spectral_centroid(audio, sr):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=min(2048, len(audio)))
    # print(centroid.mean())
    return centroid.mean()

def spectral_bandwidth(audio, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=min(2048, len(audio)))
    # print(bandwidth.mean())
    return bandwidth.mean()

def spectral_roll_off(audio, sr):
    roll_off = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=min(2048, len(audio)))
    # print(roll_off.mean())
    return roll_off.mean()

def zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=min(2048, len(audio)))
    # print(zcr.mean())
    return zcr.mean()

def mel_frequency_cepstral_coefficients(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_fft=min(2048, len(audio)))
    # print(mfccs.mean(axis=1))
    return mfccs.mean(axis=1)

def extract_features(audio, sr):
    features = []
    features.append(mean_of_chroma_vector(audio, sr))
    features.append(root_mean_quare_energy(audio, sr))
    features.append(spectral_centroid(audio, sr))
    features.append(spectral_bandwidth(audio, sr))
    features.append(spectral_roll_off(audio, sr))
    features.append(zero_crossing_rate(audio))
    features.extend(mel_frequency_cepstral_coefficients(audio, sr))
    # print(features)
    return features

def load_training_data():
    chewing = []
    swallowing = []
    drinking = []
    speaking = []
    chewing_folder_path = './data/chewing'
    swallowing_folder_path = './data/swallowing'
    drinking_folder_path = './data/drinking'
    speaking_folder_path = './data/speaking'
    
    for root, dirs, files in os.walk(chewing_folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file = os.path.join(root, file_name)
                audio, sr = librosa.load(file, sr=None)
                chewing.append(extract_features(audio, sr))
            
    for root, dirs, files in os.walk(swallowing_folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file = os.path.join(root, file_name)
                audio, sr = librosa.load(file, sr=None)
                swallowing.append(extract_features(audio, sr))
    
    for root, dirs, files in os.walk(drinking_folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file = os.path.join(root, file_name)
                audio, sr = librosa.load(file, sr=None)
                drinking.append(extract_features(audio, sr))
    
    for root, dirs, files in os.walk(speaking_folder_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file = os.path.join(root, file_name)
                audio, sr = librosa.load(file, sr=None)
                speaking.append(extract_features(audio, sr))
            
    data = chewing + swallowing + drinking + speaking
    label = [0] * len(chewing) + [1] * len(swallowing) + [2] * len(drinking) + [3] * len(speaking)
    
    with open('data.txt', 'w') as f:
        f.write(str(data))
    with open('label.txt', 'w') as f:
        f.write(str(label))
            
    return data, label

def classify(data, label):
    smo = SVMSMOTE(k_neighbors=4)
    data, label = smo.fit_resample(data, label)
    
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=1)
    svm = SVC()
    svm.fit(data_train, label_train)
    label_pred = svm.predict(data_test)
    print("-----------------SVM-----------------")
    print("predict: ", list(label_pred))
    print("actual:  ", label_test)
    print("accuracy: ", accuracy_score(label_test, label_pred))
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data_train, label_train)
    label_pred = knn.predict(data_test)
    print("-----------------KNN-----------------")
    print("predict: ", list(label_pred))
    print("actual:  ", label_test)
    print("accuracy: ", accuracy_score(label_test, label_pred))
    
    clf = DecisionTreeClassifier()
    clf.fit(data_train, label_train)
    label_pred = clf.predict(data_test)
    print("-----------------Decision Tree-----------------")
    print("predict: ", list(label_pred))
    print("actual:  ", label_test)
    print("accuracy: ", accuracy_score(label_test, label_pred))
    
def segment_data(filename):
    audio, sr = librosa.load(filename, sr=None)
    frame_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    energy = np.array([sum(abs(audio[i:i+frame_length]**2)) for i in range(0, len(audio), hop_length)])
    energy = np.log(energy + 1e-10)
    # for en in energy:
    #     print(en)
    theta = np.log(1e-5 + 1e-10)
    print(theta)
    threshold = theta
    tf = 0
    segments = []
    segments_begin = []
    begin_end = []
    always_larger = True
    for i in range(0, len(energy)):
        if energy[i] < np.log(1e-6):
            continue
        if i == 0:
            if energy[i] > threshold:
                tf = 0
                segments_begin.append(0)
            else:
                continue
        if energy[i] < threshold:
            always_larger = False
            if energy[i - 1] > threshold:
                if i * hop_length + frame_length - segments_begin[-1] > 0.6 * sr and tf * hop_length != begin_end[-1][1]:
                    segments.append(audio[(tf * hop_length) : (i * hop_length)])
                    begin_end.append((tf * hop_length, i * hop_length))
                    segments_begin.append(tf * hop_length)
                    threshold = theta
                else:
                    threshold = np.max(energy[tf : i]) - np.log(2)
                    print("threshold: ", threshold, i * hop_length / sr)
                    tf = i
                    if (len(segments) > 0):
                        segments.pop()
                    segments.append(audio[segments_begin[-1] : (i * hop_length)])
                    if (len(begin_end) > 0):
                        begin_end.pop()
                    begin_end.append((segments_begin[-1], i * hop_length))
            else:
                if len(segments_begin) == 0:
                    continue
                if (i - 1) * hop_length + frame_length - segments_begin[-1] > 0.6 * sr:
                    threshold = theta
        else:
            if len(segments_begin) == 0:
                segments_begin.append(i * hop_length)
            if energy[i - 1] < threshold:
                tf = i
    if always_larger:
        segments.append(audio[segments_begin[-1]:])
        begin_end.append((segments_begin[-1], len(audio)))
    print([seg / sr for seg in segments_begin])
    print([(t[0] / sr, t[1] / sr) for t in begin_end])
    return segments
    
def main():
    with open('data.txt', 'r') as f:
        data = json.loads(f.read())
    with open('label.txt', 'r') as f:
        label = json.loads(f.read())
    
    classify(data, label)
    
# read the audio file, segment it (?) and extract features for each segment
def test():
    with open('data.txt', 'r') as f:
        data = json.loads(f.read())
    with open('label.txt', 'r') as f:
        label = json.loads(f.read())

    segments = segment_data('./data/240307_e2.wav')
    data_test = [extract_features(segment, 48000) for segment in segments]        
    
    smo = SVMSMOTE(k_neighbors=2)
    data_train, label_train = smo.fit_resample(data, label)
    
    # data_train, data_valid, label_train, label_valid = train_test_split(data_train, label_train, test_size=0.2, random_state=1)
    
    svm = SVC()
    svm.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    label_pred = svm.predict(copy.deepcopy(data_test))
    print("svm predict: ", list(label_pred))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    label_pred = knn.predict(copy.deepcopy(data_test))
    print("knn predict: ", list(label_pred))
    
    clf = DecisionTreeClassifier()
    clf.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    label_pred = clf.predict(copy.deepcopy(data_test))
    print("clf predict: ", list(label_pred))

test()

# main()

# split_data()
# load_training_data()