import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import pylab as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from astroNN.models import Galaxy10CNN

hdulist = fits.open('G:/galaxy_AI/spectra_train_data/output/train_data_10.fits')

flux = hdulist[0].data
objid = hdulist[1].data['objid']
labels = hdulist[1].data['label']

labels = utils.to_categorical(labels, 3)
labels = labels.astype(np.float32)
flux = flux.astype(np.float32)

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
train_flux, train_labels, test_flux, test_labels = flux[train_idx], labels[train_idx], flux[test_idx], labels[test_idx]

train_flux = train_flux.reshape((-1, 3000, 1))
test_flux = test_flux.reshape((-1, 3000, 1))

model = Sequential([
    Conv1D(16, 3, activation='relu', input_shape=(3000, 1)),#kernel_initializer='random_normal'
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),#全连接层
    Dense(3, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',#优化器，SGD为梯度下降
              loss='categorical_crossentropy',#多分类 交叉熵
              metrics=['accuracy'])

# 假设 train_flux, train_labels, test_flux, test_labels 已经准备好
# 训练模型
history = model.fit(train_flux, train_labels,
                    epochs=5,
                    batch_size=32,#可以设越大越好64，128，256...迭代多少样本做一次更新
                    validation_data=(test_flux, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_flux, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")
model.summary()

predict_hdu = fits.open('G:/galaxy_AI/spectra_train_data/output/test_data.fits')
predict_flux = predict_hdu[0].data
predict_objid = predict_hdu[1].data['objid']
predict_flux = predict_flux.astype(np.float32)
predict_flux = predict_flux.reshape((-1, 3000, 1))

predictions = model.predict(predict_flux)
predict_labels = np.argmax(predictions, axis=1)
# print(f"Predicted classes: {predict_labels}")

predict_objid = predict_objid.astype(np.int)
predict_labels = predict_labels.astype(np.int)
df = pd.DataFrame({
    'obj_id': predict_objid,
    'predicted_label': predict_labels+1
})

# df.to_csv('自主命题：基于深度学习的Lamost DR9光谱分类结果.csv', index=False)
