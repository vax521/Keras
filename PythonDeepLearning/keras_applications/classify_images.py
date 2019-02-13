from keras.applications import resnet50
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

model = resnet50.ResNet50(weights='imagenet')
image_path = "../image/penguin_toy.jpg"
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predicted = model.predict(x)
print("Predicted:", decode_predictions(predicted))
