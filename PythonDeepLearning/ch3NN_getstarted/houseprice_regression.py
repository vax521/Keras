import numpy as np
from keras import layers
from keras import models
from keras.datasets import boston_housing
# import myutils.plot_utils as plot_utils
import myutils.performance_utils as performance_utils
import matplotlib.pyplot as plt
performance_utils.opitimize_cpu()

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# 因为需要将同一个模型多次实例化，所以用一个函数来构建模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K折交叉验证
k = 4
num_of_samples = len(train_data)//k
num_epochs = 100
all_scores = []
all_mae_histories = []
for i in range(k):
    print("processing fold # {}\n".format(i))
    val_data = train_data[i*num_of_samples:(i+1)*num_of_samples]
    val_targets = train_targets[i*num_of_samples:(i+1)*num_of_samples]
    partial_train_data = np.concatenate([train_data[:i*num_of_samples], train_data[(i+1)*num_of_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_of_samples], train_targets[(i+1)*num_of_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(average_mae_history)
plt.show()
