import yaml
import os
import argparse
import random
from PIL import  Image
import lib.global_val as GLV
from PyQt5.QtWidgets import QWidget, QApplication, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import time
from PyQt5.QtWidgets import QFileDialog
#获取标签名并写入列表函数
def get_classes(self,classes_path):
    with open(classes_path, encoding='gbk') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

    # 修改yaml文件函数（Cfg）
def updata_cfgyaml(self, k, v):
    self.cfg = './models/' + self.intial_weight.currentText() + '.yaml'
    with open(self.cfg) as f:
        old_data = yaml.load(f, Loader=yaml.FullLoader)
        old_data[k] = v
    with open(self.cfg, "w", encoding="GBK") as f:
        yaml.dump(old_data, f)

    # 修改yaml文件函数（opt）
def updata_optyaml(self, k, v):
    self.cfg = './runs/train/exp/opt.yaml'
    with open(self.cfg) as f:
        old_data = yaml.load(f, Loader=yaml.FullLoader)
        old_data[k] = v
    with open(self.cfg, "w", encoding="GBK") as f:
        yaml.dump(old_data, f)

    #from skimage import io#使用IO库读取tif图片

class CalculatorThread(QThread):
    signal_progress_update = pyqtSignal(list)
    signal_done=pyqtSignal(int)
    def __init__(self):
        super(CalculatorThread, self).__init__()
        self.total = 100
        self.newProject_fileName=None
        self.data_fileName=None
        self.trainVal=None
        self.train=None
    def import_datatest(self,Filename, dataFileName, Trainval_percent=0.9, Train_percent=0.9):
            parser = argparse.ArgumentParser()
            # 需要分类的图片类型
            parser.add_argument('--type', default='.tiff', type=str, help='input oring label path')
            # 原始文件的地址，存放图片和txt文档
            parser.add_argument('--orin_path', default=dataFileName, type=str, help='input oring label path')
            # 保存文件夹
            parser.add_argument('--datasets_path', default='./datasets', type=str, help='output image label path')
            # 保存图片的地址
            parser.add_argument('--Images_path', default='./datasets/' + Filename + '/images', type=str,
                                help='output image label path')
            # 保存txt的地址
            parser.add_argument('--labels_path', default='./datasets/' + Filename + '/labels', type=str,
                                help='output txt label path')

            # xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
            parser.add_argument('--xml_path', default='./datasets/' + Filename + '/images', type=str,
                                help='input xml label path')
            # 数据集的划分，地址选择自己数据下的ImageSets/Main
            parser.add_argument('--txt_path', default='./datasets/' + Filename + '/ImageSets/Main', type=str,
                                help='output txt label path')
            opt = parser.parse_args()
            type = opt.type
            orin_path = opt.orin_path
            datasets_path = opt.datasets_path
            Images_path = opt.Images_path
            lables_path = opt.labels_path

            if not os.path.exists(datasets_path):
                os.makedirs(datasets_path)
            if not os.path.exists(Images_path):
                os.makedirs(Images_path)
            if not os.path.exists(lables_path):
                os.makedirs(lables_path)

            # 保存图片
            def save_image(input_path, out_path):
                img = Image.open(input_path)
                img.save(out_path)  ## 存入的图像image

            # 保存txt文档
            def save_txt(input_path, out_path):
                # 2.调用函数进行复制
                f1 = open(input_path, mode='r', encoding='GBK')  # 用读取方式打开test1.txt文本
                f2 = open(out_path, mode='w', encoding='GBK')  # 用写入方式打开test2.txt文本 test2.txt是需要复制的文件夹
                content = f1.read()
                f2.write(content)
                f1.close()  # 关闭文件
                f2.close()
                return

            fileDir = orin_path  # 填写要读取图片文件夹的路径
            imageDir = Images_path  # 填写保存随机读取图片文件夹的路径
            labelDir = lables_path
            img_total = []
            txt_total = []
            total = os.listdir(fileDir)
            for subdir in os.listdir(fileDir):
                if subdir.endswith('.txt'):
                    txt_total.append((subdir.split('.')[0]) + type)
            # 获取图片数量,保存全局变量
            total_image = len(txt_total) - 1
            # GLV.set_value_1('total_imge', total_image)
            nuI = 0
            for subdir in os.listdir(fileDir):
                subdir_path = os.path.join(fileDir, subdir)  # 输入图片路径
                out_path_image = os.path.join(imageDir, subdir)  # 保存图片路径
                out_path_txt = os.path.join(labelDir, subdir)  # 保存txt路径
                if os.path.isfile(subdir_path):
                    if (subdir_path.endswith(type) or subdir_path.endswith('.jpg')) and subdir in txt_total:
                        if not os.path.exists(out_path_image):
                            save_image(subdir_path, out_path_image)
                            nuI = nuI+1
                            self.signal_progress_update.emit([nuI, total_image])
                            # GLV.set_value_1('newImage_total',nuI)
                    if subdir_path.endswith('.txt'):
                        if not os.path.exists(out_path_txt):
                            save_txt(subdir_path, out_path_txt)
            print("数据copy完毕")
            # 数据集划分
            trainval_percent = Trainval_percent
            train_percent = Train_percent
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
            project_path = os.path.abspath('./')

            fileName_label = './datasets/%s/labels/' % (Filename)
            for image_set in sets:
                if not os.path.exists(fileName_label):
                    os.makedirs(fileName_label)
                image_ids = open(
                    './datasets/' + Filename + '/imageSets/Main/%s.txt' % (image_set)).read().strip().split()
                list_file = open('./datasets/' + Filename + '/%s.txt' % (image_set), 'w')
                for image_id in image_ids:
                    list_file.write(
                        project_path + os.path.join('\datasets\{0}\images\{1}\n'.format(Filename, image_id)))
                list_file.close()
            print("数据分类完毕")
    def run(self):
        self.import_datatest(self.newProject_fileName,self.data_fileName,self.trainVal,self.train)
        self.signal_done.emit(1)

