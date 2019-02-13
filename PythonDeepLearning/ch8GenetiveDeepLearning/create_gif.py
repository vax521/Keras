# from PIL import Image, ImageSequence
# import os
# img_path = './image/nerual_style/'
# imgs = []
# for img_name in os.listdir(img_path):
#     img = Image.open(img_path+img_name)
#     imgs.append(img)
# imgs[0].save('./image/gif/out.gif', save_all=True, append_images=imgs[1:])


import imageio
import  os
#
# def create_gif(image_list, gif_name):
#     frames = []
#     for image_name in image_list:
#         frames.append(imageio.imread(image_name))
#     # Save them as frames into a gif
#     imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)
#
#     return
#
# def main():
#     image_list = ['test_gif-0.png', 'test_gif-2.png', 'test_gif-4.png',
#                   'test_gif-6.png', 'test_gif-8.png', 'test_gif-10.png']
#     gif_name = 'created_gif.gif'
#     create_gif(image_list, gif_name)
#

def create_gif(image_path, gif_name):
    frames=[]
    for img_name in os.listdir(image_path):
        print(image_path+img_name)
        frames.append(imageio.imread(image_path+img_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=2)

if __name__ == "__main__":
    create_gif('./image/nerual_style/', './image/gif/style.gif')