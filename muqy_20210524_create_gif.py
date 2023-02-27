# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:19:03 2020

@author: Mu o(*￣▽￣*)ブ
"""
import os
import imageio


def create_gif(image_list, gif_name):

    frames = []
    for image_name in image_list:
        if image_name.endswith(".png"):
            print(image_name)
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, "GIF", duration=0.4)

    return


def main():

    path = "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse/"  # 存放PNG图片文件夹位置
    files = os.listdir(path)
    # files.sort(key = lambda x:int(x[-11:-10]))
    files.sort(key=lambda x: int(x[-7:-5]))

    image_list = [path + img for img in files]
    gif_name = "EOFCld_gif.gif"  # 生成gif的名称
    create_gif(image_list, gif_name)


main()
