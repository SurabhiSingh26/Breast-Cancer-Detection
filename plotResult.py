import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import numpy
def Draw (clf, pred, arr, features):
    if len(features) == 2:
        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1.0
        h = .005
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.pcolormesh(xx, yy, z, cmap='tab20')
    plot_label = ["Benign", "Malignant"]
    colors = ["r", "g", "k"]
    markers = ["*", "o", "s"]
    x = [arr[i, 0] for i in range(len(pred)) if (pred[i] == 1)]
    y = [arr[i, 1] for i in range(len(pred)) if (pred[i] == 1)]
    plt.scatter(x, y, color=colors[1], marker=markers[1], label=plot_label[1])
    x = [arr[i, 0] for i in range(len(pred)) if (pred[i] == 0)]
    y = [arr[i, 1] for i in range(len(pred)) if (pred[i] == 0)]
    plt.scatter(x, y, color=colors[0], marker=markers[0], label=plot_label[0])
    plt.legend()
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()