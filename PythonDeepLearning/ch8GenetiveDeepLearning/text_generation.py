import random
import keras
import sys
from keras import layers
import numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

# 充分使用CPU
config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                        allow_soft_placement=True, device_count={'CPU': 4})
session = tf.Session(config=config)
KTF.set_session(session)


text = open("../dataset/nietzsche.txt").read().lower()
print('Corpus length:', len(text))

""""
要提取长度为 maxlen 的序列（这些序列之间存在部分重叠），对它们进行one-hot编码，
然后将其打包成形状为 (sequences, maxlen, unique_characters)的三维Numpy 数组。
"""
# 字符向量序列化
maxlen = 60  # 提取60个字符组成的序列
step = 3  # 每三个字符采样成一个新序列
sentences = []  # 保存所提取的序列
next_chars = []  # 保存目标字符

for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i+maxlen])
print("number of sentences:", len(sentences))

chars = sorted(list(set(text)))  # 语料中唯一字符组成的列表
print("The number of unique chars:", len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

print("Vectorization.....")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 构建模型
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars),activation="softmax"))
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 给定模型预测，采样下一个字符的函数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 文本生成循环
for epoch in range(1, 60):
    print('epoch:'+str(epoch)+'\n')
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]
        generated_text += next_char
        generated_text = generated_text[1:]
        sys.stdout.write(next_char)