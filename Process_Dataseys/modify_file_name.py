#### 将一个文件夹内的图片名称随机，对应着另外一个文件中的标签也改为相对应的名字
# coding=utf-8
import os  # 打开文件时需要
from PIL import Image
import re


class BatchRename():
    def __init__(self):
        # 我的图片文件夹路径
        self.path1 = '../datasets/images'
        self.path2 = '../datasets/labels'

    def rename(self):
        filelist = os.listdir(self.path1)
        total_num = len(filelist)
        print(total_num)
        i = 0  # 图片编号从多少开始
        for item in filelist:
            # if item.endswith('.png'):
            # print(i,item)
            # i = i + 1
            final_item=item.split('.')[-1]
            txt_name=item.split('.')[0]
            n = 4 - len(str(i))
            src1 = os.path.join(os.path.abspath(self.path1), item)
            dst1 = os.path.join(os.path.abspath(self.path1), 'c9_' + str(0) * n + str(i) +'.' +final_item)
            src2 = os.path.join(os.path.abspath(self.path2), txt_name+'.txt')
            dst2 = os.path.join(os.path.abspath(self.path2), 'c9_' + str(0) * n + str(i) + '.txt')
            try:

                os.rename(src1, dst1)
                os.rename(src2, dst2)
                print('converting %s to %s ...' % (src1, dst1))
                i = i + 1
            except:
                continue
        print("total %d to rename & converted %d jpgs" % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

