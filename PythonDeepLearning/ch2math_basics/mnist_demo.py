
import time
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import myutils.plot_utils as plot_utils
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 图像数据处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 标签处理
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 构建模型
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(256, activation='relu', input_shape=(28*28,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', input_shape=(28*28,)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

time_start = time.time()
history = model.fit(train_images, train_labels, epochs=80, batch_size=128,validation_data=(test_images, test_labels))
print("训练时间：{}s\n".format(int(time.time()-time_start)))
# # 分析结果
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc:{}\n".format(test_acc))

history_dict = history.history
plot_utils.plot_history(history_dict)

