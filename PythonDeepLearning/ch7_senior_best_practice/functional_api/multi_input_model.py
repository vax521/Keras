import keras
from keras.models import Model
from keras import Input
from keras import layers
import numpy as np
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_tensor = Input(shape=(None,), dtype='int32', name="text")
# 将输入嵌入64位的向量
embeded_text = layers.Embedding(text_vocabulary_size, 64)(text_tensor)
# 利用LSTM将向量编码为单个向量
embeded_text = layers.LSTM(32)(embeded_text)

question_tensotr = Input(shape=(None,), dtype='int32', name='question')
embeded_question = layers.Embedding(question_tensotr, 32)(question_tensotr)
embeded_question = layers.LSTM(16)(embeded_question)
# 将编码后的问题和文本连接起来
concatenated_tensor = layers.concatenate([embeded_text, embeded_question], axis=0)
answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated_tensor)

model = Model([text_tensor, question_tensotr], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
# 回答是 one-hot 编码的，不是整数
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)
model.fit([text, question], answers, epochs=10, batch_size=128)
# model.fit({'text': text, 'question': question}, answers,  epochs=10, batch_size=128)
