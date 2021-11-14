import cv2
import numpy as np

numberImage = cv2.imread('number.png', cv2.IMREAD_GRAYSCALE)

cells = np.hsplit(numberImage, 10)
x = np.array(cells)

train = x.reshape(-1, 140).astype(np.float32)
print(train.shape)

test = train.copy()

k = np.arange(10)
train_labels = k[:, np.newaxis]
test_labels = train_labels.copy()

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=1)

matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )

np.savez('knn_digits.npz', train = train, train_labels = train_labels)
