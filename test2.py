import os
import copy
import json
import numpy as np
import textgrid
import soundfile as sf
import librosa
import scipy
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SVMSMOTE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

DIR_NAME = './new_data/'
TRAIN_FILE_NAMES = ['0Lab', '1Lab', '2Lab', '3Res', '4Res', '5Res', '6Lab', '6Res', '7Lab', '7Res', '9Lab', '9Res', '10Res', '11Res', '12Res', '13Lab', '13Res', '14Lab', '15Res', '17Lab', '17Res']
EVALUATE_FILE_NAME = '5Res'
TIER_NAMES = {'喝液体': 'drinking', '每一口咀嚼': 'chewing', '说话': 'speaking', '吞咽': 'swallowing'}

def split_data():    
    for filename in TRAIN_FILE_NAMES:
        tg = textgrid.TextGrid()
        tg.read(DIR_NAME + filename + '.TextGrid')
        data = tg.tiers
        audio, sr = librosa.load(DIR_NAME + filename + '.wav', sr=None)
        if not os.path.exists(DIR_NAME + 'others'):
            os.makedirs(DIR_NAME + 'others')
        for tier in data:
            if tier.name != '每一口咀嚼':
                continue
            # if tier.name not in TIER_NAMES:
            #     continue
            if not os.path.exists(DIR_NAME + TIER_NAMES[tier.name]):
                os.makedirs(DIR_NAME + TIER_NAMES[tier.name])
            for interval in tier:
                if interval.mark != '':
                    print(tier.name, interval.mark, interval.minTime, interval.maxTime)
                    start_time = interval.minTime
                    end_time = interval.maxTime
                    start_frame = int(start_time * sr)
                    end_frame = int(end_time * sr)
                    if end_frame > len(audio):
                        print('end_frame > len(audio)', end_frame, len(audio))
                        continue
                    segment = audio[start_frame:end_frame]
                    sf.write(DIR_NAME + TIER_NAMES[tier.name] + '/' + filename + '_' + str(start_time) + '_' + str(end_time) + '.wav', segment, sr)
                else:
                    print(tier.name, interval.mark, interval.minTime, interval.maxTime)
                    start_time = interval.minTime
                    end_time = interval.maxTime
                    start_frame = int(start_time * sr)
                    end_frame = int(end_time * sr)
                    if end_frame > len(audio):
                        print('end_frame > len(audio)', end_frame, len(audio))
                        continue
                    segment = audio[start_frame:end_frame]
                    sf.write(DIR_NAME + 'others' + '/' + filename + '_' + str(start_time) + '_' + str(end_time) + '.wav', segment, sr)
                
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
    chewing_folder_path = DIR_NAME + 'chewing'
    swallowing_folder_path = DIR_NAME + 'swallowing'
    drinking_folder_path = DIR_NAME + 'drinking'
    speaking_folder_path = DIR_NAME + 'speaking'
    others_folder_path = DIR_NAME + 'others'
    
    folder_paths = [chewing_folder_path, swallowing_folder_path, drinking_folder_path, speaking_folder_path, others_folder_path]
    data_set = [[], [], [], [], []]
    
    for i in range(len(folder_paths)):
        for root, dirs, files in os.walk(folder_paths[i]):
            for file_name in files:
                if file_name.endswith('.wav'):
                    file = os.path.join(root, file_name)
                    audio, sr = librosa.load(file, sr=None)
                    data_set[i].append(extract_features(audio, sr))
                
    data = data_set[0] + data_set[1] + data_set[2] + data_set[3] + data_set[4]
    label = [0] * len(data_set[0]) + [1] * len(data_set[1]) + [2] * len(data_set[2]) + [3] * len(data_set[3]) + [4] * len(data_set[4])
    
    with open('test_data.txt', 'w') as f:
        f.write(str(data))
    with open('test_label.txt', 'w') as f:
        f.write(str(label))
            
    return data, label

def classify(data, label):
    # smo = SVMSMOTE(k_neighbors=2)
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
    
def segment_data(filename, theta, TIME_LAPSE):
    audio, sr = librosa.load(filename, sr=None)
    frame_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    # cutoff_freq = get_cutoff_freq(audio, sr)
    # TIME_LAPSE = 1 / cutoff_freq
    print("TIME_LAPSE: ", TIME_LAPSE)
    
    # filter_order = 2
    # B, A = scipy.signal.butter(filter_order, 200 * 2 / sr)
    # audio_filter = np.abs(scipy.signal.filtfilt(B, A, audio))
    # sf.write('filtered.wav', audio_filter, sr)
    # audio = np.abs(scipy.signal.filtfilt(B, A, audio))
    # plt.plot([i for i in range(len(audio_filter))], audio_filter)
    # plt.show()
    
    energy = np.array([sum(abs(audio[i:i+frame_length]**2)) for i in range(0, len(audio), hop_length)])
    print(len(energy))
    # energy_filter = np.array([sum(abs(audio_filter[i:i+frame_length]**2)) for i in range(0, len(audio_filter), hop_length)])
    # energy = energy_filter
    # median = np.median(energy_filter)
    # energy_satisfy = np.array([energy_filter[i] for i in range(len(energy_filter)) if energy_filter[i] > median])
    theta = np.log(np.quantile(energy, 0.925))
    # theta = np.log(energy.mean())
    energy = np.log(energy)
    print(theta)
    threshold = theta
    tf = 0
    segments = []
    
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
                if end_time_n - seg[-1][0] > TIME_LAPSE * sr:
                    threshold = theta
                    seg.append((begin_time_n, end_time_n))
                    # tf = i
                else:
                    last_seg = seg.pop()
                    seg.append((last_seg[0], end_time_n))
                    threshold = np.max(energy[tf : i]) - np.log(2)
                    # tf = i
                    # if threshold != theta:
                    #     last_seg = seg.pop()
                    #     seg.append((last_seg[0], end_time_n))
                    #     threshold = theta
                    #     # tf = i
                    # else:
                    #     threshold = np.max(energy[tf : i]) - np.log(2)
                    #     seg.append((begin_time_n, end_time_n))
                    #     # tf = i
            else:
                if end_time_n - seg[-1][0] > TIME_LAPSE * sr:
                    threshold = theta
        else:
            if energy[i - 1] < threshold:
                tf = i
            
    for s in seg:
        if s[0] > len(audio) or s[1] > len(audio):
            seg.remove(s)
            continue
        segments.append(audio[s[0]:s[1]])
    
    # print(len(audio) / sr)
    # print([(t[0] / sr, t[1] / sr) for t in seg])
    print(len(seg))
    return seg, segments, sr
    
def train():
    with open('test_data.txt', 'r') as f:
        data = json.loads(f.read())
    with open('test_label.txt', 'r') as f:
        label = json.loads(f.read())
    
    classify(data, label)
    
def get_origin_time_slots(filename):
    tg = textgrid.TextGrid()
    tg.read(DIR_NAME + filename + '.TextGrid')
    time_slots_chewing = []
    time_slots = []
    for i in range(len(tg.tiers)):
        tier = tg.tiers[i]
        if tg.tiers[i].name == '每一口咀嚼':
            for j in range(len(tier)):
                interval = tier[j]
                if interval.mark != '':
                    time_slots_chewing.append((interval.minTime, interval.maxTime, i))
                    time_slots.append((interval.minTime, interval.maxTime, i))
        else:
            for j in range(len(tier)):
                interval = tier[j]
                if interval.mark != '':
                    time_slots.append((interval.minTime, interval.maxTime, i))
    time_slots.sort(key=lambda x: x[0])
    time_slots_chewing.sort(key=lambda x: x[0])
    return time_slots, time_slots_chewing

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
            if len(origin_labels) == 0:
                FP += 1
            elif len(origin_labels) == 1:
                TP += 1
            elif len(origin_labels) > 1:
                if 0 in origin_labels:
                    FP += 1
                else:
                    all_same = True
                    target_label = origin_labels[0]
                    for label in origin_labels:
                        if label != target_label:
                            all_same = False
                            break
                    if all_same:
                        TP += 1
                    else:
                        FP += 1
                    
    print("TP: ", TP)
    print("FP: ", FP)
    return TP, FP

def get_cutoff_freq(y, sr):
    y_power_spectrum = np.abs(np.fft.fft(np.array(y)))
    y_power_spectrum = y_power_spectrum[:len(y_power_spectrum) // 2]
    y_N = len(y)
    n1 = int(2.5 * y_N / sr)
    n2 = int(5.0 * y_N / sr)
    local_power = y_power_spectrum[n1:n2]
    if len(local_power) == 0:
        return 0.0
    x_axis = np.arange(2.5, 5, (5-2.5)/len(local_power))
    if len(x_axis) != len(local_power):
        x_axis = x_axis[:len(local_power)]
    ave = np.average(local_power)
    # print(ave)
    # peak_id, _ = scipy.signal.find_peaks(local_power, height=ave)
    # peak_id, _ = scipy.signal.find_peaks(local_power, distance=int((n2-n1)/(5-2.5)))
    peak_id = np.argmax(local_power)
    # plt.plot(x_axis, local_power)
    # plt.plot(x_axis[peak_id], local_power[peak_id], "x")
    # plt.show()
    # if len(peak_id) == 0:
    #     return 0.0
    # cutoff_freq = x_axis[peak_id[0]]
    cutoff_freq = x_axis[peak_id]
    print("cutoff_freq: ", cutoff_freq)
    return cutoff_freq

def classify_models(model_name, data_train, label_train, data_test, theta):
    model = None
    if model_name == 'svm':
        model = SVC()
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == 'clf':
        model = DecisionTreeClassifier()
    model.fit(copy.deepcopy(data_train), copy.deepcopy(label_train))
    label_pred = model.predict(copy.deepcopy(data_test))
    print(model_name + " predict: ", label_pred.shape, list(label_pred))
    print(model_name, label_pred[label_pred == 0].shape)
    with open(str(theta) + 'result.txt', 'a') as f:
        f.write(model_name + ": " + str(label_pred[label_pred == 0].shape) + "\n")
    return label_pred

def write_text_grid(filename, min_time, max_time, seg, label_pred, sr, model_name, theta):
    tg = textgrid.TextGrid(minTime=min_time, maxTime=max_time)
    tier_chewing = textgrid.IntervalTier(name="chewing", minTime=min_time, maxTime=max_time)
    tier_swallowing = textgrid.IntervalTier(name="swallowing", minTime=min_time, maxTime=max_time)
    tier_drinking = textgrid.IntervalTier(name="drinking", minTime=min_time, maxTime=max_time)
    tier_speaking = textgrid.IntervalTier(name="speaking", minTime=min_time, maxTime=max_time)
    tier_others = textgrid.IntervalTier(name="others", minTime=min_time, maxTime=max_time)
    tier_chewing.strict = False
    tier_swallowing.strict = False
    tier_drinking.strict = False
    tier_speaking.strict = False
    tier_others.strict = False
    
    for i in range(len(seg)):
        interval = textgrid.Interval(minTime=seg[i][0]/sr, maxTime=seg[i][1]/sr, mark=str(label_pred[i]))
        if seg[i][1] / sr > max_time:
            continue
        if label_pred[i] == 0:
            tier_chewing.addInterval(interval)
        elif label_pred[i] == 1:
            tier_swallowing.addInterval(interval)
        elif label_pred[i] == 2:
            tier_drinking.addInterval(interval)
        elif label_pred[i] == 3:
            tier_speaking.addInterval(interval)
        elif label_pred[i] == 4:
            tier_others.addInterval(interval)
            
    tg.tiers.append(tier_chewing)  
    tg.tiers.append(tier_swallowing)
    tg.tiers.append(tier_drinking)
    tg.tiers.append(tier_speaking)
    tg.tiers.append(tier_others)
    
    tg.write(filename + "-result.TextGrid")
    
# read the audio file, segment it (?) and extract features for each segment
def test(theta, filename, TIME_LAPSE):
    # with open('test_data.txt', 'r') as f:
    #     data = json.loads(f.read())
    # with open('test_label.txt', 'r') as f:
    #     label = json.loads(f.read())

    # smo = SVMSMOTE(k_neighbors=2)
    # data_train, label_train = smo.fit_resample(data, label)
    # idx = [index for index, value in enumerate(label) if value == 0 or value == 4]
    # data_train = copy.deepcopy([data[i] for i in idx])
    # label_train = copy.deepcopy([label[i] for i in idx])
        
    # with open(str(theta) + 'result.txt', 'w') as f:
    #     f.write(str(theta) + '\n')
    
    seg, segments, sr = segment_data(DIR_NAME + filename + '.wav', theta, TIME_LAPSE)
    with open('result.txt', 'a') as f:
        f.write(filename + "\n")
        f.write("seg: " + str(len(seg)) + "\n")
    # data_test = [extract_features(segment, sr) for segment in segments]
    
    # train_len = len(data_train)
    # total_data = data_train + data_test
    # scaler = MinMaxScaler()
    # total_data = scaler.fit_transform(total_data)
    # for i in range(len(total_data)):
    #     total_data[i][1] *= 100
    # data_train = total_data[:train_len]
    # data_test = total_data[train_len:]
    
    # with open('normalized_data.txt', 'w') as f:
    #     f.write(str(data_train))
    
    # data_train, data_valid, label_train, label_valid = train_test_split(data_train, label_train, test_size=0.2, random_state=1)

    tg0 = textgrid.TextGrid()
    tg0.read(DIR_NAME + filename + '.TextGrid')
    min_time = tg0.minTime
    max_time = tg0.maxTime
    
    write_text_grid(filename, min_time, max_time, seg, [0] * len(seg), sr, str(TIME_LAPSE) + '&', theta)
    
    original_time_slots, original_time_slots_chewing = get_origin_time_slots(filename)
    label_pred = [0] * len(seg)
    TP, FP = compare(original_time_slots, seg, label_pred, sr)
    TP_chewing, FP_chewing = compare(original_time_slots_chewing, seg, label_pred, sr)
    
    with open('result.txt', 'a') as f:
        f.write("all length: " + str(len(original_time_slots)) + "\n")
        f.write("all TP: " + str(TP) + "\n")
        f.write("all FP: " + str(FP) + "\n")
        f.write("all precision: " + str(TP / (TP + FP)) + "\n")
        f.write("all recall: " + str(TP / len(original_time_slots)) + "\n")
        f.write("chewing length: " + str(len(original_time_slots_chewing)) + "\n")
        f.write("chewing TP: " + str(TP_chewing) + "\n")
        f.write("chewing FP: " + str(FP_chewing) + "\n")
        f.write("chewing precision: " + str(TP_chewing / (TP_chewing + FP_chewing)) + "\n")
        f.write("chewing recall: " + str(TP_chewing / len(original_time_slots_chewing)) + "\n")
        f.write("\n")
    
    # original_time_slots = get_origin_time_slots()
    # print(original_time_slots)
    
    # svm_pred = classify_models('svm', data_train, label_train, data_test, theta)
    # write_text_grid(min_time, max_time, seg, svm_pred, sr, 'svm', theta)
    
    # knn_pred = classify_models('knn', data_train, label_train, data_test, theta)
    # write_text_grid(min_time, max_time, seg, knn_pred, sr, 'knn', theta)
    
    # clf_pred = classify_models('clf', data_train, label_train, data_test, theta)
    # write_text_grid(min_time, max_time, seg, clf_pred, sr, 'clf', theta)
    

def main():
    # split_data()
    # load_training_data()
    # train()
    # for TIME_LAPSE in [0.25, 0.28, 0.3, 0.35, 0.4]:
        # for theta in [-2.45, -2.5, -2.53, -2.55, -2.57, -2.6, -2.7]:
    for name in TRAIN_FILE_NAMES:
        print(name)
        test(0, name, 0.3)
    
main()