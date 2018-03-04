from plotResult import Draw
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from dataextract import preprocess, datasplit
feature_mean, feature_se, feature_max, label, feature_names = preprocess()
percentile = 0.2
selected_features = feature_names
#selected_features = ["radius","texture"]
features_list = []
for i in selected_features:
    features_list.append(feature_names.index(i))
feature_mean_selected = feature_mean[:, features_list]
feature_train, feature_test, label_train, label_test = datasplit(feature_mean_selected, label, percentile)
clf = SVC(kernel='rbf', gamma=1, C=2)
clf = clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
acc = accuracy_score(label_test, pred)
print("Accuracy of this algorithm is : ", acc)
Draw(clf, pred, feature_test, selected_features)
