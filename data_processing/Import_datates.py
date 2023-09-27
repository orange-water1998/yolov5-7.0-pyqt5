
import os,sys
import argparse
from PIL import  Image
import cv2
import numpy as np
from skimage import io#使用IO库读取tif图片

parser = argparse.ArgumentParser()
#需要分类的图片类型
parser.add_argument('--type', default='.tiff', type=str, help='input oring label path')
#原始文件的地址，存放图片和txt文档
parser.add_argument('--orin_path', default='G:\隐裂数据集\隐裂测试训练\图片\不规则隐裂', type=str, help='input oring label path')
#保存文件夹
parser.add_argument('--datasets_path', default='../datasets', type=str, help='output image label path')
#保存图片的地址
parser.add_argument('--Images_path', default='../datasets/images', type=str, help='output image label path')
#保存txt的地址
parser.add_argument('--labels_path', default='../datasets/labels', type=str, help='output txt label path')
opt = parser.parse_args()
type = opt.type
orin_path = opt.orin_path
datasets_path=opt.datasets_path
Images_path = opt.Images_path
lables_path = opt.labels_path
if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)
if not os.path.exists(Images_path):
    os.makedirs(Images_path)
if not os.path.exists(lables_path):
    os.makedirs(lables_path)

#保存图片
def save_image(input_path,out_path):
    img = Image.open(input_path)
    img.save(out_path) ## 存入的图像image

#保存txt文档
def save_txt(input_path,out_path):
    # 2.调用函数进行复制
    f1 = open(input_path, mode='r', encoding='GBK')  # 用读取方式打开test1.txt文本
    f2 = open(out_path, mode='w', encoding='GBK')  # 用写入方式打开test2.txt文本 test2.txt是需要复制的文件夹
    content = f1.read()
    f2.write(content)
    f1.close()  # 关闭文件
    f2.close()
    return

if __name__ == '__main__':
    fileDir = orin_path # 填写要读取图片文件夹的路径
    imageDir = Images_path  # 填写保存随机读取图片文件夹的路径
    labelDir =lables_path
    img_total=[]
    txt_total=[]
    total=os.listdir(fileDir)
    for subdir in os.listdir(fileDir):
        if subdir.endswith('.txt'):
            txt_total.append((subdir.split('.')[0])+type)

    for subdir in os.listdir(fileDir):
        subdir_path = os.path.join(fileDir,subdir)#输入图片路径
        out_path_image = os.path.join(imageDir,subdir)#保存图片路径
        out_path_txt = os.path.join(labelDir,subdir)#保存txt路径
        if os.path.isfile(subdir_path):
            if (subdir_path.endswith(type) or subdir_path.endswith('.jpg')) and subdir in txt_total:
                if not os.path.exists(out_path_image):
                    save_image(subdir_path , out_path_image)
            if subdir_path.endswith('.txt'):
                if not os.path.exists(out_path_txt):
                    save_txt(subdir_path,out_path_txt)





