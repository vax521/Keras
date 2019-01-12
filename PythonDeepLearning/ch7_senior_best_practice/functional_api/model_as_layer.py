from keras import layers
from keras import applications
from keras import Input


xception_base = applications.Xception(weights=None, include_top=False)
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
# 图像处理基础模型是Xception 网络（只包括卷积基）
left_features = xception_base(left_input)
right_features = xception_base(right_input)
merged_features = layers.concatenate([left_features, right_features], axis=-1)