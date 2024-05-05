# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# import everything we need first
from tensorflow.keras import utils
import numpy as np
from sklearn.model_selection import train_test_split
import pylab as plt

from astroNN.models import Galaxy10CNN
from astroNN.datasets import galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# To load images and labels (will download automatically at the first time)
# First time downloading location will be ~/.astroNN/datasets/
images, labels = galaxy10.load_data()

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# # Select 10 of the images to inspect
# img = None
# plt.ion()
# print("===================Data Inspection===================")
# for counter, i in enumerate(range(np.random.randint(0, labels.shape[0], size=10).shape[0])):
#     img = plt.imshow(images[i])
#     plt.title(f"Class {np.argmax(labels[i])}: {galaxy10cls_lookup(labels[i])} \n Random Demo images {counter+1} of 10")
#     plt.draw()
#     plt.pause(2.)
# plt.close("all")
# print("===============Data Inspection Finished===============")

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

for i in range(images.shape[0]):
    img = images[i] # 转换数据类型以便进行计算
    min_val = np.min(img)               # 计算当前图像的最小值
    max_val = np.max(img)               # 计算当前图像的最大值
    images[i] = (img - min_val) / (max_val - min_val)# 执行归一化并保存结果

# Split the dataset into training set and testing set
train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=train_images.shape[1:],kernel_initializer='random_normal'),  #kernel_initializer='random_normal'
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=12, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# After the training, you can test the neural net performance
# Please notice predicted_labels are labels predicted from neural network. test_labels are ground truth from the dataset
predicted_labels = model.predict(test_images)

# Convert predicted_labels to class
prediction_class = np.argmax(predicted_labels, axis=1)

# Convert test_labels to class
test_class = np.argmax(test_labels, axis=1)

# Prepare a confusion matrix
confusion_matrix = np.zeros((10,10))

# create the confusion matrix
for counter, i in enumerate(prediction_class):
    confusion_matrix[i, test_class[counter]] += 1

# Plot the confusion matrix
galaxy10_confusion(confusion_matrix)

model.summary()