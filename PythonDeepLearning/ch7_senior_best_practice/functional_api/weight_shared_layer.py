from  keras import layers
from keras import Input
from keras.models import Model

lstm  = layers.LSTM(32)

# 左分支：输入是长度是128向量组成的变长向量
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

# 构建右分支，如果调用已有的层实例，则会共享权重
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged_output = layers.concatenate([left_output, right_output], axis=-1)
# 构建一个分类器
predictions = layers.Dense(1, activation='sigmoid')(merged_output)
model = Model([left_input, right_input], predictions)
# model.fit([left_data, right_input],targets)