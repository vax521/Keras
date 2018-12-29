from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM
import myutils.plot_utils as plot_utils
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()
max_features = 10000
maxlen = 500
batch_size = 32

print("loading data .......")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print("Pad sequences......")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model = models.Sequential()
model.add(Embedding(max_features, 32))
# model.add(SimpleRNN(32))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

results = model.evaluate(x_test, y_test)
print("results:{}\n".format(results))
plot_utils.plot_history(history.history)

