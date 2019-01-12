"""构建具有多个输出（或多头）的模型。
一个简单的例子就是一个网络试图同时预测数据的不同性质，比如一个网络，输入某个匿名人士
的一系列社交媒体发帖，然后尝试预测那个人的属性，比如年龄、性别和收入水平
"""
import keras
from keras.models import Model
from keras import Input
from keras import layers
import numpy as np
import myutils.performance_utils as performance_utils

performance_utils.opitimize_cpu()

vocabulary_size = 50000
num_income_groups = 10

post_input = Input(shape=(None,), dtype='int32', name='posts')
embeded_posts = layers.Embedding((256, vocabulary_size))(post_input)
x = layers.Conv1D(128, 5, activation='relu')(embeded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x) # 输出层都有名字
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='income')(x)

model = Model(post_input, [age_prediction, income_prediction, gender_prediction])
"""
训练这种模型需要能够对网络的各个头指定不同的损失函数，例如，年龄预测
是标量回归任务，而性别预测是二分类任务，二者需要不同的训练过程。但是，梯度下降要求
将一个标量最小化，所以为了能够训练模型，我们必须将这些损失合并为单个标量。合并不同
损失最简单的方法就是对所有损失求和。在 Keras 中，你可以在编译时使用损失组成的列表或
字典来为不同输出指定不同损失，然后将得到的损失值相加得到一个全局损失，并在训练过程
中将这个损失最小化。
"""
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# 输出层有名字时的写法
# model.compile(optimizer='rmsprop',loss={'age': 'mse','income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})