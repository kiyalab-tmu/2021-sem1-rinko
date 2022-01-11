#!/usr/bin/env python
# coding: utf-8



# In[]:
import sys
import datetime
import hashlib
from os import path


"""
ファイル内容取得
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import keras
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, AveragePooling2D, Reshape, Lambda, MaxPooling2D
from keras.models import Model
from keras import regularizers


ex_name = sys.argv[1]
gpu_id = 0
print(tf.__version__)
if tf.__version__ >= "2.1.0":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
elif tf.__version__ >= "2.0.0":
    #TF2.0
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
else:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=str(gpu_id), # specify GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))




# In[ ]:
"""
データロード
"""
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_data = train_images
test_data  = test_images
# train_data = train_data[:, :, :, np.newaxis]
# test_data  = test_data[:, :, :, np.newaxis]

print(train_data.shape)


# In[4]:
"""
ラベルの生成
"""
def createOneHotVector(labels):
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)
    label_encoded = label_encoded[:, np.newaxis]
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_vector = one_hot_encoder.fit_transform(label_encoded)
    return one_hot_vector

train_one_hot_vector = createOneHotVector(train_labels)
test_one_hot_vector = createOneHotVector(test_labels)






# In[3]:
"""
モデルのコンパイル
"""
# ###アプリ
# model1 = keras.applications.vgg16.VGG16(include_top=False, weights=None, pooling='avg',input_shape=(28,28,1))
# x = model1.output
# predictions = Dense(train_one_hot_vector.shape[1], activation='softmax')(x)
# model = Model(inputs=model1.input, outputs=predictions)
# ###アプリ

# #Lenet
# model = keras.Sequential([
#     Input(shape=(28,28)),
#     Lambda(lambda x: tf.expand_dims(x, -1)),
#     Conv2D(6, (5, 5), activation='relu', strides=(1, 1), padding='same',input_shape=(28, 28, 1)),
#     AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
#     Conv2D(16, (5, 5), activation='relu', strides=(1, 1), padding='valid'),
#     AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
#     Flatten(),
#     Dense(150, activation='relu'),
#     Dense(10, activation='softmax')
# ])
# #Lenet

#Alex
# model = tf.keras.Sequential([
#     Input(shape=(28,28)),
#     Lambda(lambda x: tf.expand_dims(x, -1)),
#     tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation='bilinear'),
#     Conv2D(96, (11, 11), activation='relu', strides=(4, 4), padding='valid'),
#     MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format=None),
#     Conv2D(256, (5, 5), activation='relu', strides=(2, 2), padding='same'),
#     MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None),
#     Conv2D(384, (3, 3), activation='relu', strides=(1, 1), padding='same'),
#     Conv2D(384, (3, 3), activation='relu', strides=(1, 1), padding='same'),
#     Conv2D(384, (3, 3), activation='relu', strides=(1, 1), padding='same'),
#     MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None),
#     Flatten(),
#     Dense(4096, activation='relu'),
#     Dropout(0.5),
#     Dense(4096, activation='relu'),
#     Dropout(0.5),
#     Dense(1000, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#Alex


# #VGG
model = tf.keras.Sequential([
    Input(shape=(28,28)),
    Lambda(lambda x: tf.expand_dims(x, -1)),
    tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation='bilinear'),
    Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
    Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
    Conv2D(256, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(256, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(256, (1, 1), activation='relu', strides=(1,1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
    Conv2D(512, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(512, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(256, (1, 1), activation='relu', strides=(1,1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
    Conv2D(512, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(512, (3, 3), activation='relu', strides=(1,1), padding='valid'),
    Conv2D(256, (1, 1), activation='relu', strides=(1,1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax')
])
# #VGG

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['acc'])
print(model.summary())
print('最終的な入力shape:', train_data.shape)












# In[ ]:
"""
学習
"""
checkpoint = keras.callbacks.ModelCheckpoint(ex_name + 'model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
history = model.fit(x=train_data, y=train_one_hot_vector, batch_size=256, epochs=50, verbose=2, validation_split=0.2, shuffle=True, callbacks=[checkpoint])









# In[]:
"""
ラストエポック後の結果
"""
print('loss:', history.history['loss'][-1])
print('val_loss:', history.history['val_loss'][-1])
print('acc:', history.history['acc'][-1])
print('val_acc:', history.history['val_acc'][-1])








# In[ ]:
"""
学習結果のグラフ表示
"""
#loss
plt.figure(figsize=(8,8))
plt.title('Loss Value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()
plt.savefig("loss_result" + ".png")
plt.clf()


#acc
plt.figure(figsize=(8,8))
plt.title('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.xlim(0, 100) 
plt.ylim(0, 1.0)
plt.show()
plt.savefig("acc_result" + ".png")
plt.clf()









# In[ ]:
"""
最良モデルでのテスト
"""
#一番良かったモデルを読み込む
model = None
model = keras.models.load_model(ex_name + 'model.h5')

#評価
test_loss, test_acc = model.evaluate(test_data, test_one_hot_vector, verbose=0)
print('\n\n最良モデルで評価\n')
print('test_loss:', test_loss)
print('test_acc:', test_acc)

train_loss, train_acc = model.evaluate(train_data, train_one_hot_vector, verbose=0)
print('\n\n最良モデルで評価\n')
print('train_loss:', train_loss)
print('train_acc:', train_acc)






# In[]:
"""
confusion matrixを生成
"""
predictions = model.predict(test_data) #softmax出力のまま。各クラスに属する確率
predictions = np.argmax(predictions, axis=1) #最大確率の要素番号を返す
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(test_labels)
label_encoded = label_encoded[:, np.newaxis]
one_hot_encoder = OneHotEncoder(sparse=False)
test_one_hot_vector = one_hot_encoder.fit_transform(label_encoded)
y_test = one_hot_encoder.inverse_transform(test_one_hot_vector) #one hotをもとに戻す

#答えが楽器Aのとき、間違って選ばれた割合recall
cm_yoko_percent = confusion_matrix(y_test, predictions)
cm_yoko_percent = np.array(cm_yoko_percent, dtype='f4')
print(cm_yoko_percent.shape)
for i in range(cm_yoko_percent.shape[0]):
    row_sum = np.sum(cm_yoko_percent[i])
    for j in range(cm_yoko_percent.shape[1]):
        if row_sum == 0:
            cm_yoko_percent[i][j] = 0
        else:
            cm_yoko_percent[i][j] = cm_yoko_percent[i][j] / row_sum * 100


label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(test_labels)
#recallを保存
sns.heatmap(cm_yoko_percent, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.savefig(ex_name +"model_recall_result" + ".png")
plt.clf()



