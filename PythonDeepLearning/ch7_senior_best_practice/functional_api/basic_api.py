from keras.models import Model, Sequential
from keras import  Input
from keras import layers
from keras import optimizers
import numpy as np
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()
# Sequential Model
Seq_model = Sequential()
Seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
Seq_model.add(layers.Dense(32, activation='relu'))
Seq_model.add(layers.Dense(10, activation='softmax'))

print("Seq_model.summary:\n", Seq_model.summary())

# 对应的函数式API实现
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
print("model.summary():\n", model.summary())

model.compile(optimizer=optimizers.RMSprop(),  loss='categorical_crossentropy')

# 模拟数据
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, batch_size=128, epochs=128)
score = model.evaluate(x_train, y_train)
print("score:\n", score)