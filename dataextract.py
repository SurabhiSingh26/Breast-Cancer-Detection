import numpy as np
import numpy
def preprocess():
    file_addr ='wdbc.data.txt'
    arr = []
    file1 = open(file_addr, "r")
    for line in file1.readlines():
        lis = np.array(line.strip().split(","))
        if lis[1] == 'B':
            lis[1] = '0'
        else:
            lis[1] = '1'
        lis = lis.astype(np.float)
        arr.append(lis)
    arr = numpy.matrix(arr)

    label = arr[:, 1]
    label = label.astype(np.int)
    feature_mean = arr[:, 2:12]
    feature_se = arr[:, 12:22]
    feature_max = arr[:, 22:32]
    feature_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",
                     "concave_points", "symmetry", "fractional_dimension"]
    return feature_mean, feature_se, feature_max, label, feature_names

def datasplit(arr,label,percentile):
    to_length = int(len(arr)*percentile)
    features_test = arr[0:to_length, :]
    features_test = features_test/features_test.max(axis=0)
    label_test = label[0:to_length]
    features_train = arr[to_length:len(arr), :]
    features_train = features_train/features_train.max(axis=0)
    label_train = label[to_length:len(arr)]
    return features_train, features_test, label_train, label_test
