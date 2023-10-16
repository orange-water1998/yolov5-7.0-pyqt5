from lib.Mainwindow import Ui_MainWindow
from lib.train_Mainwindow import Ui_Train_MainWindow
from lib.establish_Window import establish_MainWindow
from lib.import_data_Window import  import_MainWindow
from lib.dataProcessing_function import CalculatorThread
from lib.delete_data_Window  import Delete_MainWindow
from lib.share import SI
from lib.train_function import get_classes
from PyQt5.QtWidgets import QFileDialog
import lib.global_val as GLV
import sys ,yaml
import os
import train
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette,QPixmap,QBrush,QPainter
from PyQt5 import QtWidgets


class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
                #super关键字解决多继承问题
        super(MainWindow,self).__init__(parent)
        # self.ui=QUiLoader().load('uidemo.ui')

        self.setupUi(self)
        #loadUi('Mainwindow.ui')

        #导入已有训练工程
        self.action_import.triggered.connect(self.show_import_data_ui)
        #新建工程界面
        self.action_new.triggered.connect(self.show_new_data_ui)
        #删除工程
        self.action_delete.triggered.connect(self.show_delete_data_ui)
        # 检测界面
        #self.btn_detect.triggered.connect(self.show_de_ui)        #绑定点击事件
        # 模型训练界面
        #self.btn_train.triggered.connect(self.show_tr_ui)  #点击事件
        # 数据标注
        #self.btn_labeling.triggered.connect(self.labelsign)       #点击事件

    #导入已有训练工程界面
    def show_import_data_ui(self):
        SI.import_ui = Import_Window()
        SI.import_ui.show()
        SI.import_ui.exec_()
    #显示新建界面
    def show_new_data_ui(self):
        SI.new_data_ui = NewData_Window()
        SI.new_data_ui.show()
        SI.new_data_ui.exec_()
    def show_delete_data_ui(self):
        SI.delete_data_ui = Delete_Window()
        SI.delete_data_ui.show()
        SI.delete_data_ui.exec_()
    # 显示检测界面
    # def show_de_ui(self):
    #     self.detect_ui = detect_ui() #导入class detect_ui
    #     self.detect_ui.show()        #显示
    #     self.detect_ui.exec_()       #保留界面直到人为关闭
    #
    # 显示训练界面
    def show_tr_ui(self):
        self.train_ui = Train_Window()
        self.train_ui.show()
        #self.train_ui.exec_()   #//这个对话框就是模态对话框

    #弹出数据标注界面
    def labelsign(self):
        labelimg_path = os.path.dirname(os.path.realpath(sys.argv[0]))+'/utils/labelImg.exe'  #得到labelImg.exe的绝对路径
        #print(labelimg_path)
        sys.path.append(labelimg_path)  #将labelImg.exe加入到环境变量，在py文件转为exe后，可以在cmd中找到
        os.system(labelimg_path)        #利用cmd命令行打开labelImg

class Train_Window(QMainWindow,Ui_Train_MainWindow):
    def __init__(self,parent=None):
                #super关键字解决多继承问题
        super(Train_Window,self).__init__(parent)
        # self.ui=QUiLoader().load('uidemo.ui')
        #loadUi('train_MainWindow.ui')
        self.setupUi(self)
        self.thread=[] #线程数量
        self.patience=None
        self.workers =8
        self.resume=False
        self.optimizer="SGD"
        self.name='exp'
        self.batch_size = 16  #batchsize初始化
        self.weight = None #模型初始化
        self.cfg = None  #cfg初始化
        self.data = None #数据data初始化
        self.classes = [] #类别名称列表
        # self.is_train_ing = True #是否训练变量
        self.epoches = 300 #epoch初始化
        self.resume_weight = None #重新训练模型选择初始化
        self.datapath= None #数据路径初始化
        self.train_cont = 1 #1时触发初始训练，2时触发继续训练
        self.open_board = True #多线程中用于判定是否打开tensorboard，True时while循环一直触发，直到满足条件打开tensorboard后变为Flase
        self.stop_train = False #是否停止训练，按下终止检测时，会变为True_stop,而后跳过后面所有epoch，训练完当前epoch后保存结果
        self.init_My()
        self.get_model_parameters()
        self.pushButton_final.clicked.connect(self.close)
        self.pushButton_start.clicked.connect(self.train_model)
        self.QPushButton_weight.clicked.connect(self.QPushButton_weight_click_My)
        self.pushButton_suggest.clicked.connect(self.pushButton_suggest_click_My)
        self.QComboBox_suggest.currentIndexChanged.connect(self.pushButton_suggest_click_My)
        self.QComboBox_background.currentIndexChanged.connect(self.QComboBox_background_change)
        self.QpushButton_background.clicked.connect(self.QComboBox_background_change)
        # self.pushButton_stop.clicked.connect(self.pushButton_stop_Click_My)#提前停止训练
        # self.pushButton_continue.clicked.connect(self.pushButton_continue_clicked_my)#继续上次训练
        self.QPushButton_opentensorboard.clicked.connect(self.QPushButton_opentensorboard_click_my)
        self.QComboBox_background.addItems(["icon/休闲阶梯.jpeg", "icon/垂柳光影.jpeg", "icon/夕阳光影.jpeg",
                                                   "icon/微白墙.jpeg", "icon/浅蓝天空.jpg", "icon/粉白装饰镜.jpeg",
                                                   "icon/路灯光影.jpeg"])
    def init_My(self):
        self.spinBox_batch_size.setRange(4, 200)
        self.spinBox_batch_size.setValue(16)  # 设置当前值；
        self.spinBox_batch_size.setSingleStep(2)
        self.spinBox_worker.setRange(0, 40)
        self.spinBox_worker.setValue(8)  # 设置当前值；
        self.spinBox_worker.setSingleStep(1)
        self.spinBox_patience.setRange(50, 2000)
        self.spinBox_patience.setSingleStep(10)
        self.spinBox_patience.setValue(100)  # 设置当前值；
        self.spinBox_epoch.setRange(50, 214748364)
        self.spinBox_epoch.setSingleStep(10)
        self.spinBox_epoch.setValue(300)  # 设置当前值；
        self.QComboBox_suggest.addItems(["配置1","配置2","配置3"])
        self.lineEdit_weight.setText(os.path.abspath('./yolov5s.pt'))
        # self.pushButton_stop.setEnabled(False)#提前停止训练按钮冻结
        # self.pushButton_continue.setEnabled(False)  # 激活继续训练按钮

   #设置图片背景
    def QComboBox_background_change(self):
        backround_dir = self.QComboBox_background.currentText()
        palette = QPalette()
        palette.setBrush(QPalette.Window,QBrush(QPixmap(backround_dir)))
        self.setPalette(palette)
    #背景图片自适应窗口
    def paintEvent(self, event):
        painter = QPainter(self)
        self.background_image = self.QComboBox_background.currentText()
        pixmap = QPixmap(self.background_image)  ## ""中输入图片路径
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    #获取输入模型参数
    def get_model_parameters(self):

        # projectDiretory=GLV.get_value_1("projectDiretory")#获取项目路径
        self.name=GLV.get_value_1("ProjectFilename")
        self.cfg=GLV.get_value_1("cfg")#获取cfg路径
        self.data = GLV.get_value_1("yaml_path")  # 获取数据的yaml路径
        self.weight=self.lineEdit_weight.text()
        self.batch_size=self.spinBox_batch_size.value()
        self.epoches=self.spinBox_epoch.value()
        self.patience=self.spinBox_patience.value()
        self.workers=self.spinBox_worker.value()
        self.optimizer=self.QComboBox_optimize.currentText()
        name=self.name
        data=self.data
        cfg=self.cfg
        weight = self.weight
        batch_size = self.batch_size
        epochs = self.epoches
        patience=self.patience
        workers=self.workers
        optimizer=self.optimizer
        resume=self.resume
        parameters={}
        parameters["name"] = name
        parameters["cfg"] = cfg
        parameters["data"]=data
        parameters["weight"] = weight
        parameters["batch_size"] = batch_size
        parameters["epochs"] = epochs
        parameters["patience"] = patience
        parameters["workers"] = workers
        parameters["optimizer"] = optimizer
        parameters["resume"] = resume
        return  parameters
    #使用推荐参数配置
    def pushButton_suggest_click_My(self):
        suggest_item=self.QComboBox_suggest.currentText()
        self.QComboBox_suggest.currentIndexChanged.disconnect()#
        if suggest_item=="配置1":
            self.batchsize=16
            self.spinBox_batch_size.setValue(self.batchsize)
            self.workers=6
            self.spinBox_worker.setValue(self.workers)
            self.epoches=400
            self.spinBox_epoch.setValue(self.epoches)
            self.patience=100
            self.spinBox_patience.setValue(self.patience)
            self.optimizer="SGD"
            self.QComboBox_optimize.setCurrentText(self.optimizer)
        if suggest_item=="配置2":
            self.batchsize = 32
            self.spinBox_batch_size.setValue(self.batchsize)
            self.workers = 3
            self.spinBox_worker.setValue(self.workers)
            self.epoches = 500
            self.spinBox_epoch.setValue(self.epoches)
            self.patience = 100
            self.spinBox_patience.setValue(self.patience)
            self.optimizer = "Adam"
            self.QComboBox_optimize.setCurrentText(self.optimizer)
        if suggest_item=="配置3":
            self.batchsize=4
            self.spinBox_batch_size.setValue(self.batchsize)
            self.workers=4
            self.spinBox_worker.setValue(self.workers)
            self.epoches=600
            self.spinBox_epoch.setValue(self.epoches)
            self.patience = 100
            self.spinBox_patience.setValue(self.patience)
            self.optimizer = "SGD"
            self.QComboBox_optimize.setCurrentText(self.optimizer)
        self.QComboBox_suggest.currentIndexChanged.connect(self.pushButton_suggest_click_My)

    #选择初始权重
    def QPushButton_weight_click_My(self):
        fname_weight = QFileDialog.getOpenFileName(self, "选择.pt", "./", "Text documents (*.pt)")
        print('fname2='+str(fname_weight))
        if fname_weight == None or not os.path.exists(str(fname_weight[0])):
           print(fname_weight)
           print("获取self.weight路径失败")
        else:
            self.lineEdit_weight.setText(str(fname_weight[0]))
            self.weight = self.lineEdit_weight.text()
            print(self.weight)
            print("获取self.weight路径成功")

    #判断训练参数是否写入
    def pretrain_file(self):
        if self.weight!=None and self.weight !="," and self.workers!=None and self.batch_size!=None \
                and self.epoches !=None and self.patience !=None and self.name!=None :
            return True
        else:
            return False
    #训练函数
    def train_model(self):
        if self.pretrain_file():  # 第一个循环判定是否选择好模型、后缀、数据
             # if self.is_train_ing:  # 第二个循环判定是变量，按下停止或者训练完之后变为Flase
             #    self.pushButton_stop.setEnabled(True)#启用提前停止训练按钮
                parameters = self.get_model_parameters()
                print(parameters)
                #开始训练
                train_THread=threading.Thread(target=train.run,kwargs=parameters)
                train_THread.setDaemon(True)#设置为守护线程（如果主线程结束了，也随之结束）
                train_THread.start()
                self.pushButton_start.setEnabled(False)
                # self.pushButton_continue.setEnabled(False)
                self.QComboBox_optimize.setEnabled(False)
                self.spinBox_worker.setEnabled(False)
                self.spinBox_epoch.setEnabled(False)
                self.spinBox_batch_size.setEnabled(False)
                self.spinBox_patience.setEnabled(False)
                self.horizontalLayout_12.setEnabled(False)
                self.pushButton_suggest.setEnabled(False)
                self.QPushButton_weight.setEnabled(False)
                self.lineEdit_weight.setEnabled(False)
                self.QPushButton_data_supply.setEnabled(False)
                self.lineEdit_weight.setEnabled(False)
                self.QComboBox_suggest.setEnabled(False)

        else:
            QMessageBox.information(self, '提示', '请先准备好数据以及训练参数')
    #提前停止训练按钮
    # def pushButton_stop_Click_My(self):
    #     stop_train = 'True_stop'  # 停止训练变量
    #     GLV.set_value_1('stop_train', stop_train)  # 赋值，在train.py 404行读取,强行退出时依旧会训练完当前epoch并保存结果
    #     self.pushButton_stop.setEnabled(False)  # 冻结提前停止训练按钮
    #     self.pushButton_continue.setEnabled(True) #激活继续训练按钮
    #继续训练按钮
    # def pushButton_continue_clicked_my(self):
    #     #判断是否保留最后一次权重
    #     self.name = GLV.get_value_1("ProjectFilename")
    #     last_dir='runs\train\{}\weights\last.pt'.format(self.name)
    #     if os.path.exists(last_dir):
    #         self.resume = True
    #     else:
    #         self.resume = False
    #     self.train_model()
    #     self.pushButton_stop.setEnabled(True)  # 激活提前停止训练按钮
    #     self.pushButton_continue.setEnabled(False)  # 激活提前停止训练按钮

    #打开tensorboad
    def QPushButton_opentensorboard_click_my(self):
        pass
    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self, "Xpath Robot", "是否要退出，退出后未保存的数据将被清除?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if (result == QtWidgets.QMessageBox.Yes):
            stop_train = 'True_stop'  # 停止训练变量
            GLV.set_value_1('stop_train', stop_train)  # 赋值，在train.py 404行读取,强行退出时依旧会训练完当前epoch并保存结果
            event.accept()
        else:
            event.ignore()



class Import_Window(QDialog,import_MainWindow):
    def __init__(self,parent=None):
        super(Import_Window,self).__init__(parent)
        self.setupUi(self)
        self.newProject_fileName=None#导入的工程名字
        self.yaml_path=None#获取数据的yaml路径
        self.projectDiretory=None#工程根目录
        self.trainpath=None#路径
        self.testpath=None#路径
        self.valpath=None#路径
        self.res=None#打印在数据的.yaml的字典
        self.classes=None#缺陷类别字典
        self.file = './datasets'#当前目录
        self.readData = self.ReadData()
        self.pushButton_cancel.clicked.connect(self.close)
        self.pushButton_ok.clicked.connect(self.on_pushButton_ok_clicked_IMport)
    #链接按钮事件
    def on_pushButton_ok_clicked_IMport(self):

        if os.path.exists(self.file) and self.QComboBox_file.currentText() !='选择导入文件夹':
            self.newProject_fileName=self.QComboBox_file.currentText()
            # print(os.path.dirname(__file__))
            #写入yaml文件
            self.get_yaml_file_directory()
            GLV.set_value_1('ProjectFilename', self.newProject_fileName)
            self.close()
            self.show_tr_ui()
        else:
            mess_box = QMessageBox(QMessageBox.Warning, '提示', '输入数据无效')
            mess_box.exec_()

    def ReadData(self):
        if os.path.exists(self.file):
            self.ageFile=os.listdir(self.file)
            self.QComboBox_file.addItems(self.ageFile)
        else:
            pass
    # 将路径、类别名称、类别数量写入yaml
    def write_yaml(self,res):
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(res, f,allow_unicode=True)
        f.close()

    def updata_optyaml(self,k, v,output,input='./models/yolov5s.yaml'):
        with open(input) as f:
            old_data = yaml.load(f, Loader=yaml.FullLoader)
            old_data[k] = v
            f.close()
        with open(output, "w", encoding="GBK") as f:
            yaml.dump(old_data, f)
            f.close()

    def get_yaml_file_directory(self):
        self.trainpath='train.txt'
        self.valpath='val.txt'
        self.testpath='test.txt'
        self.projectDiretory = (os.path.dirname(__file__) + '\datasets\%s' % (self.newProject_fileName))
        #保存全局路径
        GLV.set_value_1("projectDiretory",self.projectDiretory)
        print(self.projectDiretory)
        self.yaml_path = (self.projectDiretory+'\{0}.yaml'.format(self.newProject_fileName))
        print(self.yaml_path)
        file_class_path = self.projectDiretory + '/labels/classes.txt' # 加载标签汇总
        if os.path.exists(file_class_path) == False:
            QMessageBox.information(self, '提示', '请确保labels下有classes.txt,其记录着所有的类别名称，一个类别一行')
        else:
            with open(file_class_path, 'r') as file:
                lines = file.readlines()
                file.close()
            num_lines = len(lines)#缺陷数量
            self.classes = get_classes(file_class_path)
            self.res = {"path": self.projectDiretory, "train": self.trainpath, "test": self.testpath, "val": self.valpath, "nc": num_lines,
                        "names": self.classes}
            self.write_yaml(res=self.res)
            GLV.set_value_1("yaml_path", self.yaml_path)
            output = (self.projectDiretory + "\yolov5s.yaml")
            self.updata_optyaml(k='nc',v=num_lines,output=(self.projectDiretory+"\yolov5s.yaml"))
            GLV.set_value_1("cfg",output)
            self.pushButton_ok.setEnabled(False)
            self.pushButton_cancel.setEnabled(False)
            QMessageBox.information(self, '提示', '数据加载完毕')
    # 显示训练界面
    def show_tr_ui(self):
        SI.train_ui = Train_Window()
        SI.train_ui.show()


    #保存全局变量提供给训练界面
    def save_global_for_train(self):
        GLV.set_value_1("projectDiretory", self.projectDiretory)
        GLV.set_value_1("yaml_path", self.yaml_path)
        GLV.set_value_1('ProjectFilename', self.newProject_fileName)

class Delete_Window(QDialog,Delete_MainWindow):
    def __init__(self,parent=None):
        super(Delete_Window,self).__init__(parent)
        self.setupUi(self)
        self.thread = {}
        self.ageFile=None
        self.ageFile_final = None
        self.agediretory=None
        self.file='./datasets'
        self.readData=self.ReadData()
        self.pushButton_cancel.clicked.connect(self.close)
        self.pushButton_ok.clicked.connect(self.on_pushButton_ok_clicked1234567)


    def on_pushButton_ok_clicked1234567(self):
        if os.path.exists(self.file) and self.QComboBox_file.currentText() !='选择删除工程':
            self.ageFile_final=self.QComboBox_file.currentText()
            print(os.path.dirname(__file__))
            self.agediretory=(os.path.dirname(__file__)+'\datasets\%s' % (self.ageFile_final))
            print(self.agediretory)
            if os.path.exists(self.agediretory):
                result = QtWidgets.QMessageBox.question(self, "确认", "确认删除工程？",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if (result == QtWidgets.QMessageBox.Yes):  # 按下“yes”触发
                    self.thread[1] = threading.Thread(target=self.Delete_data(self.agediretory))
                    self.thread[1].start()
                    self.thread[1].join()
                    if not os.path.exists(self.agediretory):
                        self.ReadData()
                        mess_box=QMessageBox(QMessageBox.information, '提示', '删除完成')
                        mess_box.exec_()
                else:
                    return
        else:
            mess_box2 = QMessageBox(QMessageBox.Warning, '警告', '不存在如此文件夹')
            mess_box2.exec_()

    def ReadData(self):
        if os.path.exists(self.file):
            self.ageFile=os.listdir(self.file)
            self.QComboBox_file.addItems(self.ageFile)
            print(self.ageFile)
        else:
            pass

    def Delete_data(self,dir):
        if os.path.exists(dir):  # 如果文件存在
            if not os.path.exists(dir):
                return False
            if os.path.isfile(dir):
                os.remove(dir)
                return
            for i in os.listdir(dir):
                t = os.path.join(dir, i)
                if os.path.isdir(t):
                    self.Delete_data(t)  # 重新调用次方法
                else:
                    os.unlink(t)
            os.rmdir(dir)  # 递归删除目录下面的空文件夹
            return



class NewData_Window(QDialog,establish_MainWindow):
    def __init__(self,parent=None):
        super(NewData_Window,self).__init__(parent)
        self.setupUi(self)
        self.thread = {}
        self.newProject_fileName=None#新工程名字
        self.data_fileName=None#训练数据的文件夹
        self.classesFile=None
        self.imagesTotal=1
        self.newImage_total=0
        self.trainVal=0.9#数据集划分比例
        self.train=0.9#数据集划分比例
        self.progressBar_data.setValue(0)
        self.is_done=0#设置完成标记

        self.yaml_path = None  # 获取数据的yaml路径
        self.projectDiretory = None  # 工程根目录
        self.trainpath = None  # 路径
        self.testpath = None  # 路径
        self.valpath = None  # 路径
        self.res = None  # 打印在数据的.yaml的字典
        self.classes = None  # 缺陷类别字典
        self.file = './datasets'  # 当前目录

        self.progressBar_data.setRange(0,100)
        self.doubleSpinBox_trainVal.setValue(0.9)  # 设置当前值；
        self.doubleSpinBox_trainVal.setRange(0,1)
        self.doubleSpinBox_trainVal.setSingleStep(0.1)
        self.doubleSpinBox_train.setValue(0.9)  # 设置当前值；
        self.doubleSpinBox_train.setRange(0, 1)
        self.doubleSpinBox_train.setSingleStep(0.1)
        self.QpushButton_chosedataflieName.clicked.connect(self.on_QpushButton_chosedataflieName_clicked46424)
        self.QpushButton_choseclass.clicked.connect(self.on_QpushButton_choseclass_clicked456)
        self.pushButton_ok.clicked.connect(self.Import_data_Thrd)
        self.pushButton_cancel.clicked.connect(self.close)
    #复制数据的函数
    def Import_data_Thrd(self):
        if not self.newName_label.text() == "":
            self.newProject_fileName = self.newName_label.text()
        print(type(self.newProject_fileName))
        print("self.newProject_fileName")
        print(self.newProject_fileName)
        if self.doubleSpinBox_trainVal.value() < 1 and self.doubleSpinBox_trainVal.value() > 0:
            self.trainVal = self.doubleSpinBox_trainVal.value()
        else:
            pass
        if self.doubleSpinBox_train.value() < 1 and self.doubleSpinBox_train.value() > 0:
            self.train = self.doubleSpinBox_train.value()
        else:
            pass
        print(self.pretrain_file())
        if self.pretrain_file():
            if os.path.exists(self.data_fileName) and os.path.exists(self.classes_fileName) and self.newProject_fileName != ",":
                self.progress_data = CalculatorThread()#复制数据的子线程
                self.progress_data.newProject_fileName = self.newProject_fileName
                self.progress_data.signal_done.connect(self.callback_done)#结束信号
                self.progress_data.data_fileName = self.data_fileName
                self.progress_data.trainVal = self.trainVal
                self.progress_data.train = self.train
                self.progress_data.signal_progress_update.connect(self.update_progress)
                self.progress_data.start()
                self.pushButton_ok.setEnabled(False)
                self.pushButton_cancel.setEnabled(False)
        else:
            QMessageBox.information(self, '提示', '请输入参数')

    #更新进度条
    def update_progress(self,values):
        self.progressBar_data.setValue(int((values[0] / values[1]) * 100))
    #链接子线程发送的结束信号
    def callback_done(self, i):
        self.is_done = i
        if self.is_done == 1:
            self.messageDialog1()  #提示爬取结束
            self.get_yaml_file_directory()#写入yaml文件

            if self.pretrain_file():
                self.show_tr_ui()
            else:
                QMessageBox.information(self, '提示', '输入数据无效，请重新创建')
    #提示训练工程复制结束
    def messageDialog1(self):
        msg_box = QMessageBox(QMessageBox.Information, '通知', '训练数据爬取结束')
        self.progressBar_data.setValue(100)  # 如果爬取成功
        msg_box.exec_()
    #选择文件按钮事件
    def on_QpushButton_chosedataflieName_clicked46424(self):
        fname1=QFileDialog.getExistingDirectory(self,"选择数据文件夹",'./')
        if fname1 == None or not os.path.exists(fname1):
           pass
        else:
            self.lineEdit_datafile.setText(fname1)
            self.data_fileName = self.lineEdit_datafile.text()
            print("获取文件路径成功")
            print(self.data_fileName )

    # 选择classese文档路径事件
    def on_QpushButton_choseclass_clicked456(self):
        fname2 = QFileDialog.getOpenFileName(self, "选择classes.txt", "./", "Text documents (*.txt)")
        print('fname2='+str(fname2))
        if fname2 == None or not os.path.exists(str(fname2[0])):
           print(fname2)
        else:
            self.lineEdit_class.setText(str(fname2[0]))
            self.classes_fileName = self.lineEdit_class.text()
            print("获取clsaaes路径成功")
            print(self.classes_fileName)

    def pretrain_file(self):
        if self.newProject_fileName != None  and self.data_fileName != None   and self.classes_fileName != None and \
                os.path.exists(self.data_fileName) and os.path.exists(self.classes_fileName) and  self.newName_label.text() != "":
            return True
        else:
            return False
    #展示训练界面
    def show_tr_ui(self):
        SI.train_ui = Train_Window()
        SI.train_ui.show()


    def get_yaml_file_directory(self):
        self.trainpath='train.txt'
        self.valpath='val.txt'
        self.testpath='test.txt'
        self.projectDiretory = (os.path.dirname(__file__) + '\datasets\%s' % (self.newProject_fileName))
        #保存全局路径
        GLV.set_value_1("projectDiretory",self.projectDiretory)
        print(self.projectDiretory)
        self.yaml_path = (self.projectDiretory+'\{0}.yaml'.format(self.newProject_fileName))
        print(self.yaml_path)
        file_class_path = self.projectDiretory + '/labels/classes.txt' # 加载标签汇总
        if os.path.exists(file_class_path) == False:
            QMessageBox.information(self, '提示', '请确保labels下有classes.txt,其记录着所有的类别名称，一个类别一行')
        else:
            with open(file_class_path, 'r') as file:
                lines = file.readlines()
                file.close()
            num_lines = len(lines)#缺陷数量
            self.classes = get_classes(file_class_path)
            self.res = {"path": self.projectDiretory, "train": self.trainpath, "test": self.testpath, "val": self.valpath, "nc": num_lines,
                        "names": self.classes}
            self.write_yaml(res=self.res)
            #保存数据的.yaml文件路径
            GLV.set_value_1("yaml_path", self.yaml_path)
            output = (self.projectDiretory + "\yolov5s.yaml")
            self.updata_optyaml(k='nc',v=num_lines,output=(self.projectDiretory+"\yolov5s.yaml"))
            #保存模型的.yaml文件
            GLV.set_value_1("cfg",output)
            #保存工程名字
            GLV.set_value_1('ProjectFilename', self.newProject_fileName)
            self.pushButton_ok.setEnabled(False)
            self.pushButton_cancel.setEnabled(False)
            # QMessageBox.information(self, '提示', 'yaml文件已创建完成')

    def save_global_for_train(self):
        GLV.set_value_1("projectDiretory", self.projectDiretory)
        GLV.set_value_1("yaml_path", self.yaml_path)
        GLV.set_value_1('ProjectFilename', self.newProject_fileName)

    def write_yaml(self,res):
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(res, f,allow_unicode=True)
        f.close()

    def updata_optyaml(self,k, v,output,input='./models/yolov5s.yaml'):
        with open(input) as f:
            old_data = yaml.load(f, Loader=yaml.FullLoader)
            old_data[k] = v
            f.close()
        with open(output, "w", encoding="GBK") as f:
            yaml.dump(old_data, f)
            f.close()




if __name__ == "__main__":
    app = QApplication([])#实例化
    mainWin =MainWindow()
    mainWin.show()
    sys.exit(app.exec())
    app.exec_()


