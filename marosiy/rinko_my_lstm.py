#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tqdm import tqdm
from keras.losses import categorical_crossentropy


"""
データ読み込み
"""
ex_name = 'my_lstm'
epoch_num = 200
with open('data.txt', 'r', encoding='UTF-8') as f:
    text = f.read()

"""
データ閲覧
"""
# テキストの長さは含まれる文字数
print ('**DEBUG**' + 'Length of text: {} characters'.format(len(text)))
# テキストの最初の 250文字を参照
print('**DEBUG**' + text[:250])
# ファイル中のユニークな文字の数
vocab = sorted(set(text))
print ('**DEBUG**' + '{} unique characters'.format(len(vocab)))

"""
もじと数字を対応させる
"""
# それぞれの文字からインデックスへの対応表を作成
char2idx = {u:i for i, u in enumerate(vocab)} #{'\n': 0, ' ': 1, '!': 2, '"': 3,・・・
idx2char = np.array(vocab) # ['\n' ' ' '!' '"' '#' '$' '%' "'" '(' ')' '*' ',' '-
#単語だったら、大文字を小文字にした方がいいかもしれないけど、
#文字予測だったら、大文字小文字を分けた状態で学習した方がいいかもしれない

"""
数字化
"""
text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int.shape)

"""
学習準備
"""
# ひとつの入力としたい文字セットの長さ
seq_length = 100

seq_list = []
y = []
for i in range(0, len(text) - seq_length, 3):
    seq_list.append(text[i:i+seq_length])
    y.append(text[i+seq_length])
seq_list = np.array(seq_list)
y2 = np.array(y)


# x : np.bool型 3次元配列 [文の数, 文の最大長, 字の種類]　⇒ 文中の各位置に各indexの文字が出現するか
x = np.zeros((len(seq_list), seq_length, len(vocab)), dtype=np.bool)

# y : np.bool型 2次元配列 [文の数, 字の種類]              ⇒ 次の文の開始文字のindex
y = np.zeros((len(seq_list), len(vocab)), dtype=np.bool)

# vector化は各「文」について実施
for i, sentence in enumerate(seq_list):
    for t, char in enumerate(sentence):
        x[i, t, char2idx[char]] = 1
    y[i, char2idx[y2[i]]] = 1
# seq_list_3d = np.reshape(seq_list,(len(seq_list),seq_length,1)) #とりあえず3次元にするらしい



# y = np_utils.to_categorical(y)


"""
train test split
"""
seq_list_train, seq_list_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
def loss(y_true, y_pred):
    los = categorical_crossentropy(y_true, y_pred)
    return np.e ** los
  

"""
モデル構築
"""
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(vocab))))
model.add(Dense(y.shape[1], activation='softmax'))
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])


"""
学習
"""
checkpoint = ModelCheckpoint(ex_name + '_model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
history = model.fit(seq_list_train, y_train, batch_size=32, epochs=epoch_num, verbose=2, validation_split=0.2, shuffle=True, callbacks=[checkpoint])


model = None
model = tf.keras.models.load_model(ex_name + '_model.h5')



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
plt.savefig(ex_name+ "_loss_result" + ".png")
plt.clf()


#acc
plt.figure(figsize=(8,8))
plt.title('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.xlim(0, epoch_num) 
plt.ylim(0, 1.0)
plt.show()
plt.savefig(ex_name + "_acc_result" + ".png")
plt.clf()


test_loss, test_acc = model.evaluate(seq_list_test, y_test, verbose=0)

# 上記のランダムで選ばれた「文」に続く400個の「字」をモデルから予測し出力する
generated = text[0:0+seq_length]
sentence = text[0:0+seq_length]
for i in range(400):

    # 現在の「文」の中のどの位置に何の「字」があるかのテーブルを
    # フィッティング時に入力したxベクトルと同じフォーマットで生成
    # 最初の次元は「文」のIDなので0固定
    x_pred = np.zeros((1, seq_length, len(vocab)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char2idx[char]] = 1.

    # 現在の「文」に続く「字」を予測する
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(preds)
    next_char = idx2char[next_index]

    # 予測して得られた「字」を生成し、「文」に追加
    generated += next_char

    # モデル入力する「文」から最初の文字を削り、予測結果の「字」を追加
    # 例：sentence 「これはドイツ製」
    #     next_char 「の」
    #     ↓
    #     sentence 「れはドイツ製の」
    sentence = sentence[1:] + next_char
print(generated)

print('*********************************************')
generated = text[171000:171000+seq_length]
sentence = text[171000:171000+seq_length]
for i in range(400):

    # 現在の「文」の中のどの位置に何の「字」があるかのテーブルを
    # フィッティング時に入力したxベクトルと同じフォーマットで生成
    # 最初の次元は「文」のIDなので0固定
    x_pred = np.zeros((1, seq_length, len(vocab)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char2idx[char]] = 1.

    # 現在の「文」に続く「字」を予測する
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(preds)
    next_char = idx2char[next_index]

    # 予測して得られた「字」を生成し、「文」に追加
    generated += next_char

    # モデル入力する「文」から最初の文字を削り、予測結果の「字」を追加
    # 例：sentence 「これはドイツ製」
    #     next_char 「の」
    #     ↓
    #     sentence 「れはドイツ製の」
    sentence = sentence[1:] + next_char
print(generated)