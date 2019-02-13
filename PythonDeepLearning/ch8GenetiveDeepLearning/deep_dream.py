from keras.applications import inception_v3
import numpy as np
from keras import backend as K
import myutils.performance_utils as performance_utils
performance_utils.opitimize_cpu()
import scipy
import imageio
from keras.preprocessing import image


##工具函数
def resize_img(img, size):
     img = np.copy(img)
     factors = (1,float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
     return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(fname, pil_img)


def preprocess_image(image_path):
     img = image.load_img(image_path)
     img = image.img_to_array(img)
     img = np.expand_dims(img, axis=0)
     img = inception_v3.preprocess_input(img)
     return img


def deprocess_image(x):
     if K.image_data_format() == 'channels_first':
         x = x.reshape((3, x.shape[2], x.shape[3]))
         x = x.transpose((1, 2, 0))
     else:
         x = x.reshape((x.shape[1], x.shape[2], 3))
         x /= 2.
         x += 0.5
         x *= 255.
         x = np.clip(x, 0, 255).astype('uint8')
     return x

K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

layer_contributions = {
 'mixed2': 0.2,
 'mixed3': 3.,
 'mixed4': 2.,
 'mixed5': 1.5,
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 在定义损失时将层的贡献添加到这个标量变量中
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # 将该层特征的L2范数添加到loss中。为了避免出现边界伪影，损失中仅包含非边界的像素
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

# 设置梯度上升过程

# 这个张量用于保存生成的图像，即梦境图像
dream = model.input

grads = K.gradients(loss, dream)[0]
# 梯度标准化
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# 给定一张输出图像，设置一个 Keras 函数来获取损失值和梯度值
outputs = [loss, grads]

fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
     outs = fetch_loss_and_grads([x])
     loss_value = outs[0]
     grad_values = outs[1]
     return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x = x + step * grad_values
    return x

step = 0.01
num_octave = 3
octave_scale = 1.4
# iterations = 20
# max_loss = 10.
iterations = 30
max_loss = 15.
base_image_path = './image/demo.jpg'
img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='./image/dream_at_scale_' + str(shape) + '.png')
    save_img(img, fname='./image/final_dream.png')

