import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy
import math
import copy

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, X_train, y_train):
        """
        Train the nearest neighbor classifier.
        
        Parameters:
        - X_train: numpy array, shape (num_samples, num_features), training data.
        - y_train: numpy array, shape (num_samples,), training labels.
        """
        self.X_train = X_train
        self.y_train = y_train
            
    
    def predict(self, X_test):
        """
        Predict labels for test data using the nearest neighbor classifier.
        
        Parameters:
        - X_test: numpy array, shape (num_test_samples, num_features), test data.
        
        Returns:
        - y_pred: numpy array, shape (num_test_samples,), predicted labels for test data.
        """
        num_test = len(X_test)
        y_pred = np.zeros(num_test)
        X_test = np.asarray(X_test)
        self.X_train = np.asarray(self.X_train)
        num_1 = self.y_train.count(1)
        num_2 = self.y_train.count(2)
        num_3 = self.y_train.count(3)
        
        # Loop over all test examples
        # for i in range(num_test):
        #     # Compute the Euclidean distances between the current test example and all training examples
        #     distances = np.sqrt(np.sum(np.square(self.X_train - X_test[i]), axis=1))
        #     distance = [0.0, 0.0, 0.0, 0.0]
        #     for m in range(4):
        #         for j in range(len(self.tags[m])):
        #             sub = self.tags[m][j] - X_test[i]
        #             dis = 0.0
        #             for k in range(len(sub)):
        #                 dis += sub[k] ** 2
        #             distance[m] += dis ** 0.5
        #         distance[m] /= len(self.tags[m])
        #     print(distance)
        #     # Find the index of the training example with the smallest distance
        #     min_index = np.argmin(distance)
        #     # Predict the label of the test example based on the nearest neighbor
        #     y_pred[i] = min_index
        
        for i in range(num_test):
            # if x_test[i][0] == 0:
            #     y_pred[i] = 0
            #     continue
            # Compute the Euclidean distances between the current test example and all training examples
            distances = np.sqrt(np.sum(np.square(self.X_train - X_test[i]), axis=1))
            # Find the index of the training example with the smallest distance
            min_index = np.argmin(distances)
            # Predict the label of the test example based on the nearest neighbor
            y_pred[i] = self.y_train[min_index]
            
            # distance = [0.0, 0.0, 0.0]
            # for j in range(len(self.X_train)):
            #     dis = 0.0
            #     for k in range(len(self.X_train[j])):
            #         dis += (self.X_train[j][k] - X_test[i][k]) ** 2
            #     distance[self.y_train[j] - 1] += dis ** 0.5
            # print(distance)
            # distance[0] /= num_1
            # distance[1] /= num_2
            # distance[2] /= num_3
            # print(distance)
            # min_index = np.argmin(distance)
            # y_pred[i] = min_index + 1
                
        
        return y_pred

def load_audio(file_name):
    audio_file = file_name
    y, sr = librosa.load(audio_file, sr=48000, dtype=np.float32)
    # y = y[:48000*42]
    return y, sr

def get_features(y, sr, a):
    segments = [y[i:i+sr*a] for i in range(0, len(y), sr*a)]
    features = []
    j = 0

    for segment in segments:
        n = len(segment)
        # if n < 48000*a:
        #     continue
        power_spectrum = np.abs(np.fft.fft(np.array(segment)))
        power_spectrum = power_spectrum[:n//2]
        frequencies = np.fft.fftfreq(n, d=1/sr)
        positive_frequencies = frequencies[:n//2]
        # D = np.abs(librosa.stft(segment))
        
        start_freq = 200
        end_freq = 3000
        start_index = np.where(positive_frequencies >= start_freq)[0][0]
        end_index = np.where(positive_frequencies <= end_freq)[0][-1]
                
        positive_frequencies = positive_frequencies[start_index:end_index]
        power_spectrum = power_spectrum[start_index:end_index]
        
        if len(power_spectrum) == 0:
            continue
        
        barycentric_frequency = np.sum(np.abs(positive_frequencies) * np.abs(power_spectrum)) / np.sum(np.abs(power_spectrum))
        # barycentric_frequency = librosa.feature.spectral_centroid(S=power_spectrum, sr=sr)
        # maxium_peak_frequency = np.argmax(D, axis=0) * sr / D.shape[0]
        maxium_peak_index = np.argmax(np.abs(power_spectrum))
        maxium_peak_frequency = positive_frequencies[maxium_peak_index]
        # roll_off = librosa.feature.spectral_rolloff(y=y, sr=sr)
        energy_threshold = 0.9 * np.sum(np.abs(power_spectrum) ** 2)
        roll_off_index = np.where(np.cumsum(np.abs(power_spectrum) ** 2) >= energy_threshold)[0][0]
        roll_off_frequency = positive_frequencies[roll_off_index]
        # lpc_coefficients = np.squeeze(scipy.signal.lfilter([1], [1] + [-a for a in np.polyfit(segment, np.arange(len(segment)), 10)[1:]], segment))
        # lpc_coefficients = lpc_coefficients[:n//2]
        lpc_coefficients = librosa.lpc(segment, order=12)
                
        # center_frequencies = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        #               1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        center_frequencies = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500]
        # power_ratios = []
        # for i in range(len(center_frequencies) - 1):
        #     start_index = np.where(frequencies >= center_frequencies[i])[0][0]
        #     end_index = np.where(frequencies <= center_frequencies[i + 1])[0][-1]
        #     power_in_band = np.sum(spectrogram_data[start_index:end_index + 1, :], axis=0)
        #     power_ratio = np.sum(power_in_band) / np.sum(spectrogram_data)
        #     power_ratios.append(power_ratio)
        band_powers = []
        for i in range(len(center_frequencies) - 1):
            start_index_ = np.where(positive_frequencies >= center_frequencies[i])[0][0]
            end_index_ = np.where(positive_frequencies <= center_frequencies[i + 1])[0][-1]
            band_amplitude = np.abs(power_spectrum[start_index_:end_index_ + 1])
            band_power = np.sum(band_amplitude ** 2)
            band_powers.append(band_power)
        total_power = np.sum(np.abs(power_spectrum) ** 2)
        power_ratios = [band_power / total_power for band_power in band_powers]

        power_in_range = np.sum(np.abs(power_spectrum) ** 2)
            
        # print("Barycentric Frequency: ", barycentric_frequency)
        # print("Maxium Peak Frequency: ", maxium_peak_frequency)
        # print("Roll Off: ", roll_off_frequency)
        # print("LPC Coefficients: ", lpc_coefficients)
        # print("Power in Range: ", power_in_range)
        # j += 1
        # print("Power Ratios: ", power_ratios)
        # if len(power_spectrum) < 3000:
        #     plt.plot(np.arange(200, len(power_spectrum), 1), np.abs(power_spectrum[200:]))
        # elif len(power_spectrum) >= 3000:
        
        # plt.plot(np.arange(200, len(power_spectrum)+200, 1/a), np.abs(power_spectrum))
        # plt.plot([int(barycentric_frequency)], [np.abs(power_spectrum[(int(barycentric_frequency)-200)*a])], "x", label="Barycentric Frequency")
        # plt.plot([int(maxium_peak_frequency)], [np.abs(power_spectrum[(int(maxium_peak_frequency)-200)*a])], "x", label="Maxium Peak Frequency")
        # plt.plot([int(roll_off_frequency)], [np.abs(power_spectrum[(int(roll_off_frequency)-200)*a])], "x", label="Roll Off Frequency")
        # plt.legend()
        # plt.show()
        
        # if power_in_range < 0.5*a:
        #     features.append([0]*15)
        # else:
        
        
        feature = []
        feature.append(barycentric_frequency)
        feature.append(maxium_peak_frequency)
        feature.append(roll_off_frequency)
        feature.extend(lpc_coefficients)
        feature.append(power_in_range/a)
        feature.extend(power_ratios)
        # print(feature)
        # feature = scaler.fit_transform(np.array(feature).reshape(-1, 1)).reshape(-1)
        # print(feature)
        features.append(feature)

    return features
        

def classify(x_train, y_train, x_test):
    nn_classifier = NearestNeighbor()
    # Train the classifier
    nn_classifier.train(x_train, y_train)
    # Predict labels for test data
    y_pred = nn_classifier.predict(x_test)
    
def get_cutoff_freq(y, sr):
    y_power_spectrum = np.abs(np.fft.fft(np.array(y)))
    y_power_spectrum = y_power_spectrum[:len(y_power_spectrum) // 2]
    y_N = len(y)
    n1 = int(0.5 * y_N / sr)
    n2 = int(5.0 * y_N / sr)
    local_power = y_power_spectrum[n1:n2]
    if len(local_power) == 0:
        return 0.0
    x_axis = np.arange(0.5, 5, (5-0.5)/len(local_power))
    if len(x_axis) != len(local_power):
        x_axis = x_axis[:len(local_power)]
    ave = np.average(local_power)
    # print(ave)
    peak_id, _ = scipy.signal.find_peaks(local_power, height=ave)
    # peak_id, _ = scipy.signal.find_peaks(local_power, distance=int((n2-n1)/(5-0.5)))
    plt.plot(x_axis, local_power)
    plt.plot(x_axis[peak_id], local_power[peak_id], "x")
    # plt.show()
    if len(peak_id) == 0:
        return 0.0
    cutoff_freq = x_axis[peak_id[0]]
    print("cutoff_freq: ", cutoff_freq)
    return cutoff_freq

def count_chew(cutoff_freq, y, sr):
    filter_order = 2
    B, A = scipy.signal.butter(filter_order, cutoff_freq * 2 / sr)
    filtered = np.abs(scipy.signal.filtfilt(B, A, y))
    ave_ = np.average(filtered)
    # mastication_id, _ = scipy.signal.find_peaks(filtered, distance=int(sr/cutoff_freq/2), prominence=1e-7)
    mastication_id, _ = scipy.signal.find_peaks(filtered)
    plt.plot([i for i in range(len(filtered))], filtered)
    plt.plot(mastication_id, filtered[mastication_id], "x")
    # plt.show()
    # plt.plot([i for i in range(len(y))], y)
    # plt.show()
    print(len(mastication_id))
    return len(mastication_id)

if __name__ == '__main__':
    train_file_name_food = ["./240307/240307_e1.wav", "./240307/240307_e2.wav", "./240307/240307_e3.wav", 
                            "./240307/240307_e5.wav", "./240307/240307_e6.wav", "./240306/240306_e17.wav", 
                            "./240306/240306_e9.wav"]
    train_file_name_water = ["./240306/240306_e7.wav", "./240306/240306_e11.wav", "./240306/240306_e19.wav"]
    train_file_name_speak = ["./240306/240306_e8.wav", "./240306/240306_e12.wav", "./240306/240306_e20.wav"]
    
    x_train = []
    y_train = []
    
    for file in train_file_name_food:
        train_audio, train_sr = load_audio(file)
        x_train_ = get_features(train_audio, train_sr, 3)
        x_train.extend(x_train_)
        y_train.extend([1]*len(x_train_))
        # for i in range(len(x_train_)):
        #     if x_train_[i][0] == 0:
        #         continue
        #     else:
        #         x_train.append(x_train_[i])
        #         y_train.append(1)
        # for i in range(len(x_train_)):
        #     plt.scatter(np.arange(0, len(x_train_[0])*5, 5), x_train_[i], color="red")
        
    for file in train_file_name_water:
        train_audio, train_sr = load_audio(file)
        x_train_ = get_features(train_audio, train_sr, 3)
        x_train.extend(x_train_)
        y_train.extend([2]*len(x_train_))
        # for i in range(len(x_train_)):
        #     if x_train_[i][0] == 0:
        #         continue
        #     else:
        #         x_train.append(x_train_[i])
        #         y_train.append(2)
        # for i in range(len(x_train_)):
        #     plt.scatter(np.arange(1.5, len(x_train_[0])*5+1.5, 5), x_train_[i], color="blue")
        
    for file in train_file_name_speak:
        train_audio, train_sr = load_audio(file)
        x_train_ = get_features(train_audio, train_sr, 3)
        x_train.extend(x_train_)
        y_train.extend([3]*len(x_train_))
        # for i in range(len(x_train_)):
        #     if x_train_[i][0] == 0:
        #         continue
        #     else:
        #         x_train.append(x_train_[i])
        #         y_train.append(3)
        # for i in range(len(x_train_)):
        #     plt.scatter(np.arange(3, len(x_train_[0])*5+3, 5), x_train_[i], color="green")
    
    # plt.show()
    
    x_train0 = copy.deepcopy(x_train)
    scaler = StandardScaler()
    x_train0 = scaler.fit_transform(x_train0)
    # for i in range(len(x_train0[0])):
    #     scaler = MinMaxScaler()
    #     arg = [x_train0[j][i] for j in range(len(x_train0))]
    #     arg = scaler.fit_transform(np.array(arg).reshape(-1, 1)).reshape(-1)
    #     for j in range(len(x_train0)):
    #         x_train0[j][i] = arg[j]
    # print(x_train[0])
    # print(x_train0[0])
    X_train, X_test, Y_train, Y_test = train_test_split(x_train0, y_train)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)
    print(Y_test)
    print(Y_pred)

    test_file_names = ["./240306/240306_e5.wav", "./240306/240306_e6.wav", "./240306/240306_e7.wav", "./240306/240306_e8.wav",
                       "./240306/240306_e9.wav", "./240306/240306_e10.wav", "./240306/240306_e11.wav", "./240306/240306_e12.wav",
                       "./240306/240306_e13.wav", "./240306/240306_e14.wav", "./240306/240306_e15.wav", "./240306/240306_e16.wav",
                       "./240306/240306_e17.wav", "./240306/240306_e18.wav", "./240306/240306_e19.wav", "./240306/240306_e20.wav",
                       "./240306/240306_e21.wav", "./240306/240306_e22.wav", "./240306/240306_e23.wav", "./240306/240306_e24.wav",]
    test_file_names0 = ["./240307/240307_e1.wav", "./240307/240307_e2.wav", "./240307/240307_e3.wav", 
                        "./240307/240307_e4.wav", "./240307/240307_e5.wav", "./240307/240307_e6.wav"]
    ans = []
    for test_file_name in test_file_names:
        print(test_file_name)
        test_audio, test_sr = load_audio(test_file_name)
        x_test = (get_features(test_audio, test_sr, 1))
        
        # print(x_train[0])
        x = copy.deepcopy(x_train) + x_test
        # for i in range(len(x[0])):
        #     scaler = MinMaxScaler()
        #     arg = [x[j][i] for j in range(len(x))]
        #     arg = scaler.fit_transform(np.array(arg).reshape(-1, 1)).reshape(-1)
        #     for j in range(len(x)):
        #         x[j][i] = arg[j]
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        # print(x[0])    
        # nn_classifier = NearestNeighbor()
        # nn_classifier.train(x_train, y_train)
        # y_pred = nn_classifier.predict(x_test)
        # print(y_pred)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x[:len(x_train)], y_train)
        y_pred = knn.predict(x[len(x_train):])
        print(y_pred)
                
        chew_audios = []
        begin = 0
        flag = 0
        for i in range(len(y_pred)):
            if y_pred[i] != 1:
                if flag == 1:
                    # if i > begin + 1:
                    chew_audios.append(test_audio[begin*test_sr*1:i*test_sr*1])
                    flag = 0
                    print("range: ", begin, i)
                    # chew_audios.append(test_audio[i*test_sr*1:(i+1)*test_sr*1])
            else:
                if flag == 0:
                    begin = i
                    flag = 1
                    if i == len(y_pred) - 1 and i > begin + 1:
                        chew_audios.append(test_audio[begin*test_sr*1:])
                        print("range: ", begin, i)
                else:
                    if i == len(y_pred) - 1:
                        chew_audios.append(test_audio[begin*test_sr*1:])
                        print("range: ", begin, i)
        if len(chew_audios) == 0:
            ans.append(0)
            continue
        # chew_audio = test_audio
        chew_count = 0
        for chew_audio in chew_audios:
            # if len(chew_audio) <= test_sr*1:
            #     continue
            cutoff_freq = get_cutoff_freq(chew_audio, test_sr)
            if cutoff_freq != 0:
                chew_count += count_chew(cutoff_freq, chew_audio, test_sr)
        ans.append(chew_count)
        print("chew_count: ", chew_count)
    standard_ans = [37, 37, 0, 0, 26, 35, 0, 0, 25, 20, 0, 15, 20, 20, 0, 0, 25, 25, 0, 15]
    print("classification accuracy: ", accuracy)
    print("chew count:        ", ans)
    print("actual chew count: ", standard_ans)
