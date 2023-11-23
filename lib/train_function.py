import yaml
from PyQt5.QtWidgets import *
import lib.global_val as GLV


def pretrain_file(self):
    if self.intial_weight != None and self.datapath != None and self.batchsize != None and self.intbox_epoch != None:
        return True
    else:
        QMessageBox.information(self, '提示', '请先准备好数据以及训练参数')

def write_yaml(self,res):
    with open(self.yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(res, f,allow_unicode=True)


def stop(self):
        self.stop_train = 'True_stop'
        GLV.set_value_1('stop_train',self.stop_train)
        self.is_train_ing = False
        self.button_continnue.setEnabled(True)

def continnue(self):
        self.updata_optyaml('epochs',int(self.epoches+10))
        self.stop_train = 'True_continnue'
        GLV.set_value_1('stop_train', self.stop_train)
        self.train_cont = 2
        self.is_train_ing = True

#获取标签名并写入列表函数
def get_classes(classes_path):
    with open(classes_path, encoding='gbk') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


