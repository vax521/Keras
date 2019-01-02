import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import myutils.plot_utils as plot_utils
import myutils.performance_utils as performance_utils
performance_utils.opitimize_cpu()
f = open(r"..\dataset\jena_climate_2009_2016.csv")
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))
float_data = np.zeros((len(lines), len(header)-1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
# 数据探索
# print(float_data.shape)
# print(float_data[0])
# plt.plot(float_data[:, 1])
# plt.show()

# 数据标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def data_generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data)-delay-1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = data_generator(float_data, lookback=lookback, delay=delay, min_index=0,
                      max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = data_generator(float_data, lookback=lookback, delay=delay, min_index=200001,
                         max_index=300000, step=step, batch_size=batch_size)
test_gen = data_generator(float_data, lookback=lookback, delay=delay, min_index=300001,
                       max_index=None, step=step, batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) //batch_size
test_steps = (len(float_data) - 300001 - lookback) //batch_size


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        for samples, targets in val_gen:
            preds = samples[:, -1, 1]
            mae = np.mean(np.abs(preds - targets))
            batch_maes.append(mae)
    print(np.mean(batch_maes))
# evaluate_naive_method()


def fnn_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback//step, float_data.shape[-1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    return model


def gru_model():
    """GRU model"""
    model = models.Sequential()
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    return model


def gru_dropout_model():
    """使用Dropout正则化"""
    model = models.Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    return model


def iter_gru_model():
    """使用Dropout正则化的GRU堆叠层"""
    model = models.Sequential()
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
                         return_sequences=True, input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    return model

def get_Bidirectional_GRU():
    model = models.Sequential()
    model.add(layers.Bidirectional(layers.GRU(32),input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
model = iter_gru_model()
model.compile(optimizer=optimizers.RMSprop(), loss='mse', metrics=['mae'])
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)
plot_utils.plot_regression_his(history.history)
