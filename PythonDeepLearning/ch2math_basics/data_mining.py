from keras.datasets import mnist
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape:"+str(train_images.shape)+"\n")
print("train_images.ndim:"+str(train_images.ndim)+"\n")
print("train_images.dtype:"+str(train_images.dtype)+"\n")


# 张量切片
my_train = train_images[100:1000]
#  : 等同于选择整个轴
# 等价写法
my_train = train_images[100:1000, :, :]
my_train = train_images[100:1000, 0:28, 0:28]
print("mytrain.shape:{}\n".format(my_train.shape))
# 在所有图像的右下角选出 14 像素×14 像素的区域：
myslice = my_train[:, 14:, 14:]


random_digits = train_images[np.random.randint(5000)]
# 在图像中心裁剪出 14 像素×14 像素的区域
# random_digits = random_digits[7:-7, 7:-7]
# 在图像中心裁剪出 20 像素×20像素的区域
random_digits = random_digits[4:-4, 4:-4]
plt.imshow(random_digits, cmap=plt.cm.binary)
plt.show()