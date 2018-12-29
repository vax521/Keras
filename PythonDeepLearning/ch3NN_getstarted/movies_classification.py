from keras.datasets import imdb
from keras import layers
from keras import models
from keras import optimizers
import numpy as np
import myutils.plot_utils as plot_utils
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path="imdb.npz", num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)


# 将整数序列转换为二进制矩阵
def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
y_train = vectorize_sequences(test_data)
# 标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
model = models.Sequential()
# model.add(layers.Dense())
model.add(layers.Dense(6, activation='relu', input_shape=(10000, )))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs=5, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
results = model.evaluate(x_val, y_val)
print("results:{}\n".format(results))
print("predict:{}\n".format(model.predict(x_val)[20]))
history_dict = history.history
plot_utils.plot_history(history_dict)
