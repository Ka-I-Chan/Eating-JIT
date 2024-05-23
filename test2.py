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
    # filenames = ['240306_e5', '240306_e6', '240306_e7', '240306_e8', '240306_e9', '240306_e10', '240306_e11',
    #              '240306_e12', '240306_e13', '240306_e14', '240306_e15', '240306_e16', '240307_e1', '240307_e2',
    #              '240307_e3', '240307_e4', '240307_e5', '240307_e6', '2']
    filenames = ['2']
    
    for filename in filenames:
        tg = textgrid.TextGrid()
        tg.read('./test_data/' + filename + '.TextGrid')
        data = tg.tiers
        audio, sr = librosa.load('./test_data/' + filename + '.wav', sr=None)
        for tier in data:
            if not os.path.exists('./test_data/' + tier.name):
                os.makedirs('./test_data/' + tier.name)
            for interval in tier:
                if interval.mark != '':
                    print(tier.name, interval.mark, interval.minTime, interval.maxTime)
                    start_time = interval.minTime
                    end_time = interval.maxTime
                    start_frame = int(start_time * sr)
                    end_frame = int(end_time * sr)
                    segment = audio[start_frame:end_frame]
                    sf.write('./test_data/' + tier.name + '/' + filename + '_' + str(start_time) + '_' + str(end_time) + '.wav', segment, sr)
                
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
    chewing_folder_path = './test_data/chewing'
    swallowing_folder_path = './test_data/swallowing'
    drinking_folder_path = './test_data/drinking'
    speaking_folder_path = './test_data/speaking'
    
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
    
    with open('test_data.txt', 'w') as f:
        f.write(str(data))
    with open('test_label.txt', 'w') as f:
        f.write(str(label))
            
    return data, label

def classify(data, label):
    # smo = SVMSMOTE(k_neighbors=4)
    # data, label = smo.fit_resample(data, label)
    
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2)
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
    energy = np.log(energy)
    # for en in energy:
    #     print(en)
    theta = -5
    print(theta)
    threshold = theta
    tf = 0
    segments = []
    # segments_begin = []
    # begin_end = []
    # always_larger = True
    # for i in range(0, len(energy)):
    #     if i == 0:
    #         if energy[i] > threshold:
    #             tf = 0
    #             segments_begin.append(0)
    #         else:
    #             continue
    #     if energy[i] < threshold:
    #         always_larger = False
    #         if energy[i - 1] > threshold:
    #             if len(begin_end) > 0:
    #                 last_end = begin_end[-1][1]
    #             else:
    #                 last_end = 0
    #             if (i - 1) * hop_length - segments_begin[-1] > 0.4 * sr and tf * hop_length != last_end:
    #                 print("[1] segment: ", tf * hop_length / sr, i * hop_length / sr)
    #                 segments.append(audio[(tf * hop_length) : ((i - 1) * hop_length + frame_length)])
    #                 begin_end.append((tf * hop_length, (i - 1) * hop_length + frame_length))
    #                 segments_begin.append(tf * hop_length)
    #                 threshold = theta
    #             else:
    #                 threshold = np.max(energy[tf : i]) - np.log(2)
    #                 print("[2] threshold: ", threshold, i * hop_length / sr)
    #                 # tf = i
    #                 if (len(segments) > 0):
    #                     segments.pop()
    #                 segments.append(audio[segments_begin[-1] : ((i - 1) * hop_length + frame_length)])
    #                 if (len(begin_end) > 0):
    #                     begin_end.pop()
    #                 begin_end.append((segments_begin[-1], (i - 1) * hop_length + frame_length))
    #         else:
    #             print("[3]")
    #             if len(segments_begin) == 0:
    #                 continue
    #             if (i - 1) * hop_length - segments_begin[-1] > 0.4 * sr:
    #                 threshold = theta
    #     else:
    #         if len(segments_begin) == 0:
    #             segments_begin.append(i * hop_length)
    #         if energy[i - 1] < threshold:
    #             tf = i
    # if always_larger:
    #     segments.append(audio[segments_begin[-1]:])
    #     begin_end.append((segments_begin[-1], len(audio)))
    
    seg = []
    begin_pos = 0
    while energy[begin_pos] < threshold:
        begin_pos += 1
    tf = begin_pos
    seg.append([begin_pos * hop_length, begin_pos * hop_length + frame_length])
    for i in range(begin_pos + 1, len(energy)):
        # print(i, energy[i])
        if energy[i] < threshold:
            begin_time_n = tf * hop_length
            end_time_n = (i - 1) * hop_length + frame_length
            if energy[i - 1] > threshold:
                # print(i)
                if end_time_n - seg[-1][0] > 0.6 * sr:
                    threshold = theta
                    seg.append((begin_time_n, end_time_n))
                    tf = i
                else:
                    last_seg = seg.pop()
                    seg.append((last_seg[0], end_time_n))
                    threshold = np.max(energy[tf : i]) - np.log(2)
                    tf = i
            else:
                if end_time_n - seg[-1][0] > 0.6 * sr:
                    threshold = theta
        else:
            if energy[i - 1] < threshold:
                tf = i
            
    for s in seg:
        segments.append(audio[s[0]:s[1]])
    
    print([(t[0] / sr, t[1] / sr) for t in seg])
    return seg, segments, sr
    
def train():
    with open('test_data.txt', 'r') as f:
        data = json.loads(f.read())
    with open('test_label.txt', 'r') as f:
        label = json.loads(f.read())
    
    classify(data, label)
    
def get_origin_time_slots():
    tg = textgrid.TextGrid()
    tg.read('./test_data/2.TextGrid')
    time_slots = []
    for i in range(len(tg.tiers)):
        tier = tg.tiers[i]
        for j in range(len(tier)):
            interval = tier[j]
            if interval.mark != '':
                time_slots.append((interval.minTime, interval.maxTime, i))
    return time_slots

def compare(origin, seg, predict, sr):
    TP = 0
    FP = 0
    origin_idx = 0
    for i in range(len(predict)):
        predict_begin_time = seg[i][0] / sr
        predict_end_time = seg[i][1] / sr
        predict_label = predict[i]
        origin_labels = []
        for j in range(0, len(origin)):
            origin_begin_time = origin[j][0]
            origin_end_time = origin[j][1]
            origin_label = origin[j][2]
            if predict_begin_time <= origin_end_time and predict_end_time >= origin_begin_time:
                origin_labels.append(origin_label)
            elif predict_end_time < origin_begin_time:
                # origin_idx = j
                break
        #     print(seg[i], predict_label, origin[j])
        # print(origin_labels)
        if predict_label == 0:
            all_same = True
            for label in origin_labels:
                if label != predict_label:
                    all_same = False
                    break
            # if len(origin_labels) == 1 and origin_labels[0] == 0:
            if all_same:
                TP += 1
            else:
                FP += 1
        else:
            TP += 1
    print("TP: ", TP)
    print("FP: ", FP)
    return TP, FP
    
# read the audio file, segment it (?) and extract features for each segment
def test():
    with open('test_data.txt', 'r') as f:
        data = json.loads(f.read())
    with open('test_label.txt', 'r') as f:
        label = json.loads(f.read())

    seg, segments, sr = segment_data('./test_data/2.wav')
    data_test = [extract_features(segment, sr) for segment in segments]        
    
    # smo = SVMSMOTE(k_neighbors=2)
    # data_train, label_train = smo.fit_resample(data, label)
    data_train = copy.deepcopy(data)
    label_train = copy.deepcopy(label)
    
    # data_train, data_valid, label_train, label_valid = train_test_split(data_train, label_train, test_size=0.2, random_state=1)

    tg0 = textgrid.TextGrid()
    tg0.read('./test_data/2.TextGrid')
    min_time = tg0.minTime
    max_time = tg0.maxTime
    
    original_time_slots = get_origin_time_slots()
    print(original_time_slots)
    
    # svm = SVC()
    # svm.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    # label_pred = svm.predict(copy.deepcopy(data_test))
    # print("svm predict: ", list(label_pred))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    label_pred = knn.predict(copy.deepcopy(data_test))
    print("knn predict: ", label_pred.shape, list(label_pred))
    print("knn", label_pred[label_pred == 0].shape)
    compare(original_time_slots, seg, label_pred, sr)

    tg1 = textgrid.TextGrid(minTime=min_time, maxTime=max_time)
    print(tg1.__dict__)

    tier_chewing = textgrid.IntervalTier(name="chewing", minTime=min_time, maxTime=max_time)
    tier_swallowing = textgrid.IntervalTier(name="swallowing", minTime=min_time, maxTime=max_time)
    tier_drinking = textgrid.IntervalTier(name="drinking", minTime=min_time, maxTime=max_time)
    tier_speaking = textgrid.IntervalTier(name="speaking", minTime=min_time, maxTime=max_time)
    tier_chewing.strict = False
    tier_swallowing.strict = False
    tier_drinking.strict = False
    tier_speaking.strict = False
    
    for i in range(len(seg)):
        interval = textgrid.Interval(minTime=seg[i][0]/sr, maxTime=seg[i][1]/sr, mark=str(label_pred[i]))
        if label_pred[i] == 0:
            tier_chewing.addInterval(interval)
        elif label_pred[i] == 1:
            tier_swallowing.addInterval(interval)
        elif label_pred[i] == 2:
            tier_drinking.addInterval(interval)
        elif label_pred[i] == 3:
            tier_speaking.addInterval(interval)

    # 添加到tg对象中
    tg1.tiers.append(tier_chewing)
    tg1.tiers.append(tier_swallowing)
    tg1.tiers.append(tier_drinking)
    tg1.tiers.append(tier_speaking)

    # 写入保存
    tg1.write("knn.TextGrid")
    
    clf = DecisionTreeClassifier()
    clf.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    label_pred = clf.predict(copy.deepcopy(data_test))
    print("clf predict: ", label_pred.shape, list(label_pred))
    print("clf", label_pred[label_pred == 0].shape)
    compare(original_time_slots, seg, label_pred, sr)
    
    tg2 = textgrid.TextGrid(minTime=min_time, maxTime=max_time)
    print(tg2.__dict__)

    tier_chewing = textgrid.IntervalTier(name="chewing", minTime=min_time, maxTime=max_time)
    tier_swallowing = textgrid.IntervalTier(name="swallowing", minTime=min_time, maxTime=max_time)
    tier_drinking = textgrid.IntervalTier(name="drinking", minTime=min_time, maxTime=max_time)
    tier_speaking = textgrid.IntervalTier(name="speaking", minTime=min_time, maxTime=max_time)
    tier_chewing.strict = False
    tier_swallowing.strict = False
    tier_drinking.strict = False
    tier_speaking.strict = False
    
    for i in range(len(seg)):
        interval = textgrid.Interval(minTime=seg[i][0]/sr, maxTime=seg[i][1]/sr, mark=str(label_pred[i]))
        if label_pred[i] == 0:
            tier_chewing.addInterval(interval)
        elif label_pred[i] == 1:
            tier_swallowing.addInterval(interval)
        elif label_pred[i] == 2:
            tier_drinking.addInterval(interval)
        elif label_pred[i] == 3:
            tier_speaking.addInterval(interval)

    # 添加到tg对象中
    tg2.tiers.append(tier_chewing)
    tg2.tiers.append(tier_swallowing)
    tg2.tiers.append(tier_drinking)
    tg2.tiers.append(tier_speaking)

    # 写入保存
    tg2.write("clf.TextGrid")


def main():
    # split_data()
    # load_training_data()
    # train()
    test()
    
main()