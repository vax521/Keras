from keras.applications import resnet50
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import  Model
model = resnet50.ResNet50(weights='imagenet')
print("model.summary():", model.summary())
image_path = "../image/elephant.jpg"
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print("Features:", features)

# 从任意一个中间层提取特征

model = Model(inputs=model.input, outputs=model.get_layer('res5c_branch2b').output)

img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

res5c_branch2b_features = model.predict(x)
print("res5c_branch2b_features:",res5c_branch2b_features)