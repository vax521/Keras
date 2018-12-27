import tensorflow as tf
import time
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt

# 充分使用CPU
config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                        allow_soft_placement=True, device_count={'CPU': 4})
session = tf.Session(config=config)
KTF.set_session(session)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
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
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
print(model.summary())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

time_start = time.time()
history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))
print("训练时间：{}s\n".format(int(time.time()-time_start)))
# # 分析结果
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc:{}\n".format(test_acc))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)

plt.figure(figsize=(8, 16))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, '-', label='Train Acc')
plt.plot(epochs, val_acc, '--', label='Validation Acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

