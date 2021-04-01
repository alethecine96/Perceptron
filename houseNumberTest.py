from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle

train = sio.loadmat("train_32x32.mat")
test = sio.loadmat("test_32x32.mat")

dimensions = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
testAccuracy = []
trainAccuracy = []
labeldata = []
train_data = []
test_data = []

# Load Train Dataset

for h in range(0, 65536):
    image_gray = cv2.cvtColor(train["X"][:, :, :, h], cv2.COLOR_RGB2GRAY)
    image_gray = image_gray / 255
    train_data.append(image_gray.ravel())
labeldata = np.asarray(train['y'][:65536]).ravel()

# Load Test Dataset

for j in range(0, 26032):
    image_grayscale = cv2.cvtColor(test["X"][:, :, :, j], cv2.COLOR_RGB2GRAY)
    image_grayscale = image_grayscale / 255
    test_data.append(image_grayscale.ravel())
label_test = np.asarray(test['y']).ravel()
np.place(label_test, label_test == 10, 0)
np.place(labeldata, labeldata == 10, 0)
labeldata, train_data = shuffle(labeldata, train_data, random_state=0)

for dimension in dimensions:
    print("The Perceptron is training with " + str(dimension) + " elements")

    label = labeldata[0:dimension]
    train = train_data[0:dimension]

    clf = Perceptron(tol=1e-3, random_state=0, max_iter=1000)
    clf.fit(train, label)
    testAccuracy.append(accuracy_score(label_test, clf.predict(test_data)))
    trainAccuracy.append(accuracy_score(label, clf.predict(train)))

plt.plot(dimensions, testAccuracy, marker="o", color='blue', label='TestAccuracy')
plt.plot(dimensions, trainAccuracy, marker="o", color='green', label='TrainAccuracy')
plt.legend(loc='upper left')
plt.title("Perceptron Performance in street view house numbers classifications")
plt.xlabel("Training-Set Dimension")
plt.ylabel("Accuracy")

plt.savefig('Perceptron accuracy')

plt.show()

