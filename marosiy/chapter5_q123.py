#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt


"""
データ読み込み
"""
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

"""
数字化
"""
text_as_int = np.array([char2idx[c] for c in text])


"""
学習準備
"""
# ひとつの入力としたいシーケンスの文字数としての最大の長さ
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
# Datasetをtfフォーマットに変更
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
#サンプルをちょっと見てみる
for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
#指定した長さのシーケンスを作る
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# for item in sequences.take(5):
#     print(repr(''.join(idx2char[item.numpy()])))

#入力と答えのペアを生成
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# for input_example, target_example in  dataset.take(1):
#   print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#   print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

"""
バッチの生成
"""
# バッチサイズ
BATCH_SIZE = 64

# データセットをシャッフルするためのバッファサイズ
# （TF data は可能性として無限長のシーケンスでも使えるように設計されています。
# このため、シーケンス全体をメモリ内でシャッフルしようとはしません。
# その代わりに、要素をシャッフルするためのバッファを保持しています）
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


"""
モデルの作成
"""
# 文字数で表されるボキャブラリーの長さ
vocab_size = len(vocab)
# 埋め込みベクトルの次元
embedding_dim = 256
# RNN ユニットの数
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.SimpleRNN(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        # tf.keras.layers.GRU(rnn_units,
        #                     return_sequences=True,
        #                     stateful=True,
        #                     recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(
vocab_size = len(vocab),
embedding_dim=embedding_dim,
rnn_units=rnn_units,
batch_size=BATCH_SIZE)

model.summary()
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer='adam', loss=loss)


# チェックポイントが保存されるディレクトリ
checkpoint_dir = './training_checkpoints'
# チェックポイントファイルの名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS=500
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


#loss
plt.figure(figsize=(8,8))
plt.title('Loss Value')
plt.plot(history.history['loss'])
plt.show()
plt.savefig("word_predictor_loss_result" + ".png")
plt.clf()

"""
TEST用のモデルを構築
"""
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()



def generate_text(model, start_string):
  # 評価ステップ（学習済みモデルを使ったテキスト生成）

  # 生成する文字数
  num_generate = 1000

  # 開始文字列を数値に変換（ベクトル化）
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 結果を保存する空文字列
  text_generated = []

  # 低い temperature　は、より予測しやすいテキストをもたらし
  # 高い temperature は、より意外なテキストをもたらす
  # 実験により最適な設定を見つけること
  temperature = 1.0

  # ここではバッチサイズ　== 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # バッチの次元を削除
      predictions = tf.squeeze(predictions, 0)

      # カテゴリー分布をつかってモデルから返された文字を予測 
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 過去の隠れ状態とともに予測された文字をモデルへのつぎの入力として渡す
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string="The Project"))