from keras.utils.np_utils import  to_categorical
from keras.datasets import reuters
from keras import layers, losses
from keras import models
from keras import optimizers
import numpy as np
from keras import regularizers
import myutils.plot_utils as plot_utils
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# 将整数序列转换为二进制矩阵
def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(10000, )))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=['accuracy'])

# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_val, y_val)
print("results:{}\n".format(results))
print("predict:{}\n".format(model.predict(x_val)[20]))
plot_utils.plot_history(history.history)