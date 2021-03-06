import time

from keras import layers
from keras import models
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

import myutils.performance_utils as performance_utils
import myutils.plot_utils as plot_utils

performance_utils.opitimize_cpu()

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)

# 图像数据处理
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
# 标签处理
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.35))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
# model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

time_start = time.time()
history = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_data=(test_images, test_labels))
print("训练时间：{}s\n".format(int(time.time()-time_start)))
# # 分析结果
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc:{}\n".format(test_acc))
plot_utils.plot_history(history.history)
