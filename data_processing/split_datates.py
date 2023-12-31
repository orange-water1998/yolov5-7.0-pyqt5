# coding:utf-8

import os
import random
import argparse
from os import getcwd
parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default='../datasets22/images', type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='../datasets22/ImageSets/Main', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 0.9
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')


for i in list_index:
    name = total_xml[i] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)
file_trainval.close()
file_train.close()
file_val.close()
file_test.close()

sets = ['train', 'val', 'test']
project_path = os.path.abspath('../')
def run():
    if not os.path.exists('../datasets22/labels/'):
        os.makedirs('../datasets22/labels/')
    image_ids = open('../datasets22/imageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('../datasets22/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(project_path+'\datasets\images\%s\n' % (image_id))
    list_file.close()

for image_set in sets:
    if not os.path.exists('../datasets22/labels/'):
        os.makedirs('../datasets22/labels/')
    image_ids = open('../datasets22/imageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('../datasets22/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(project_path+'\datasets\images\%s\n' % (image_id))
    list_file.close()
