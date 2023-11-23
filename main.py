from lib.Mainwindow import Ui_MainWindow
from lib.train_Mainwindow import Ui_Train_MainWindow
from lib.establish_Window import establish_MainWindow
from lib.import_data_Window import  import_MainWindow
from lib.dataProcessing_function import CalculatorThread
from lib.delete_data_Window  import Delete_MainWindow
from lib.Tensorboard_Window import tensorboard_MainWindow
from lib.detect_window import Ui_detect_MainWindow
from lib.detect_new_window import detect_new_window
from lib.detect_import_window  import detect_import_window
from lib.detect_delete_window  import detect_delete_window
# from lib.detect_save_imgs_window import save_imgs_Win
from lib.share import SI
from lib.train_function import get_classes
from PyQt5.QtWidgets import QFileDialog
import lib.global_val as GLV
import yaml
import multiprocessing
import train
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette,QPixmap,QBrush,QPainter,QImage,QIcon
from PyQt5 import QtWidgets,QtWebEngineWidgets,QtGui
from PyQt5.QtCore import QUrl,QEvent,Qt,QThread,pyqtSignal
import webbrowser
import shutil
import numpy as np
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode



class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
                #super关键字解决多继承问题
        super(MainWindow,self).__init__(parent)
        # self.ui=QUiLoader().load('uidemo.ui')

        self.setupUi(self)
        #loadUi('Mainwindow.ui')

        #导入已有训练工程
        self.action_import.triggered.connect(self.show_import_data_ui)
        #新建训练工程界面
        self.action_new.triggered.connect(self.show_new_data_ui)
        #删除训练工程
        self.action_delete.triggered.connect(self.show_delete_data_ui)
        # 新建检测界面
        self.action1_new.triggered.connect(self.show_detect_new_ui)        #绑定点击事件
        # 模型训练界面
        self.action2_import.triggered.connect(self.show_detect_import_ui)  #点击事件
        # 删除测试工程
        self.action_delete_2.triggered.connect(self.show_detect_delete_ui)       #点击事件
        #数据标注
        self.actionsss.triggered.connect(self.start_THread)
        #帮助
        self.action_help.triggered.connect(self.action_help_click_my)
        #退出
        self.action13344.triggered.connect(self.close_program)


        backround_dir = 'icon/background.jpg'
        image = QPixmap(backround_dir)
        # 将图片缩放到窗口大小
        scaled_image = image.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(scaled_image))
        self.setPalette(palette)
        # 加载图标文件
        icon = QIcon("icon/Ai.jpg")
        # 设置窗口图标
        self.setWindowIcon(icon)
    def action_help_click_my(self):
        tip_message = "联系：2628541804@qq.com"
        QMessageBox.information(None, "提示", tip_message, QMessageBox.Ok)
    def action13344_click_my(self):
        self.close()

    def close_program(self):
        QApplication.quit()
    #导入已有训练工程界面
    def show_import_data_ui(self):
        SI.import_ui = Import_Window()
        SI.import_ui.show()
        SI.import_ui.exec_()
    #显示新建训练界面
    def show_new_data_ui(self):
        SI.new_data_ui = NewData_Window()
        SI.new_data_ui.show()
        SI.new_data_ui.exec_()
    def show_delete_data_ui(self):
        SI.delete_data_ui = Delete_Window()
        SI.delete_data_ui.show()
        SI.delete_data_ui.exec_()
    # 显示训练界面
    def show_tr_ui(self):
        self.train_ui = Train_Window()
        self.train_ui.show()
        #self.train_ui.exec_()   #//这个对话框就是模态对话框
    #新建测试界面
    def show_detect_new_ui(self):
        self.Detect_newWin = Detect_new_win()
        self.Detect_newWin.show()
        self.Detect_newWin.exec_()

    # 导入测试界面
    def show_detect_import_ui(self):
        self.Detect_importWin = Detect_import_win()
        self.Detect_importWin.show()
        self.Detect_importWin.exec_()

    #删除测试工程
    def show_detect_delete_ui(self):
        self.Detect_deleteWin = Detect_delete_win()
        self.Detect_deleteWin.show()
        self.Detect_deleteWin.exec_()
    def labelsign(self):
        labelimg_path = os.path.dirname(os.path.realpath(sys.argv[0]))+r'\utils\labelImg.exe'  #得到labelImg.exe的绝对路径
        #print(labelimg_path)
        sys.path.append(labelimg_path)  #将labelImg.exe加入到环境变量，在py文件转为exe后，可以在cmd中找到
        os.system(labelimg_path)
    def start_THread(self):
        thread_labelimg = threading.Thread(target=self.labelsign)
        thread_labelimg.start()
#检测类
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    finished = pyqtSignal(int)
    def __init__(self,):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'
        self.model=None
        self.save_to_file=False
        self.project=ROOT / 'runs/detect'
        self.is_show=True
    def __del__(self):
        self.wait()


    @smart_inference_mode()
    def run(self,
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            nosave=False,  # do not save images/videos
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride

    ):
        if self.save_to_file:
            project=self.project
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        conf_thres=self.conf_thres
        source = str(self.source)
        if self.save_to_file:
            image_extensions = ['.jpg', '.jpeg', '.tiff']  # 图片文件的扩展名
            image_count=0
            for file_name in os.listdir(source):
                file_extension = os.path.splitext(file_name)[1].lower()  # 获取文件扩展名，并转换为小写
                if file_extension in image_extensions:
                    image_count += 1
            total_image=image_count
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        bs = 1  # batch_size
        model=self.model
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # Run inference
        webcam=False
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)  # 将numpy格式图像转化成torch格式，放入gpu中
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0             #图片像素值归一化
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim,增加batch维度，一次处理多少图像文件

            # Inference
            with dt[1]:
                pred = model(im, augment=augment, visualize=visualize)  #

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1  #

                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                if self.save_to_file:
                    save_path = str(save_dir / p.name)  # im.jpg # 保存图像路径

                s += '%gx%g ' % im.shape[2:]  # print string，
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh，
                imc = im0.copy() if save_crop else im0  # for save_crop，
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                # Stream results
                im0 = annotator.result()  #
                if self.save_to_file:
                    cv2.imwrite(save_path, im0)
                    # self.signal_progress_update.emit([seen, total_image])

                # print(type(im0))
                if self.is_show:
                    self.send_img.emit(im0)
                    # self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_msg.emit(str(f'{dt[1].dt * 1E3:.1f}'))
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        self.finished.emit(1)



#加载模型类
class get_model(QThread):
    signal_done = pyqtSignal(int)
    def __init__(self,):
        super(get_model, self).__init__()
        self.weights = 'yolov5s.pt'
        self.data=''
        self.model=None
    def __del__(self):
        self.wait()
    def get_para(self,
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            dnn=False,  # use OpenCV DNN for ONNX inference
            half=False,  # use FP16 half-precision inference
            ):
        # 第三部分：加载模型的权重
        device = select_device(device)
        model = DetectMultiBackend(weights=self.weights, device=device, dnn=dnn, data=self.data, fp16=half)
        self.model = model
        self.signal_done.emit(1)


    def run(self):
        self.get_para()

# class updata_progress(QThread):
#     def updata_progress(self,values):
#         SI.save_imgs_win.progressBar.setValue(int((values[0] / values[1]) * 100))


class Detect_window(QMainWindow,Ui_detect_MainWindow):
    def __init__(self,parent=None):
                #super关键字解决多继承问题
        super(Detect_window,self).__init__(parent)
        self.setupUi(self)
        self.weights = GLV.get_value_1('detect_weight')
        self.data = GLV.get_value_1('detect_data')
        self.projectName= GLV.get_value_1('detect_ProjectFilename')
        self.source = None
        self.TotalNum = 0
        self.model=None
        self.conf_thres = 0.25
        self.image_paths = []#图片的路径集合
        self.image_path=None#图片的路径
        self.current_image_index = 1
        self.doubleSpinBox_confidence.setValue(0.25)
        self.pushButton_exeit.clicked.connect(self.close)
        self.pushButton_weight.clicked.connect(self.pushButton_weight_click_my)
        self.pushButton_dirIMage.clicked.connect(self.pushButton_dirIMage_click_my)
        self.pushButton_oneImage.clicked.connect(self.pushButton_oneImage_click_my)
        self.pushButton_next.clicked.connect(self.switch_next_image)
        self.pushButton_up.clicked.connect(self.switch_prev_image)
        self.pushButton_confidence.clicked.connect(self.pushButton_confidence_click_my)
        self.pushButton_saveImage.clicked.connect(self.pushButton_saveImage_click_my)
        self.pushButton_saveDir.clicked.connect(self.pushButton_saveDir_click_my)
        self.label_13.installEventFilter(self)
        self.label.installEventFilter(self)
        self.doubleSpinBox_confidence.setRange(0, 1)
        self.doubleSpinBox_confidence.setSingleStep(0.01)
        self.imageo=None
        # self.get_models()  # 得到模型

    #批量保存文件
    def pushButton_saveDir_click_my(self):
        if self.source!=None and os.path.exists(self.source):

            folder_path = QFileDialog.getExistingDirectory(self, "选择保存文件夹", './')
            if folder_path == None or not os.path.exists(folder_path):
                error_message = "没有选择文件。"
                QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
            else:
                result = QtWidgets.QMessageBox.question(self, "Xpath Robot", "确认，是否批量检测文件夹：" + folder_path,
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if (result == QtWidgets.QMessageBox.Yes):
                    # self.show_progress()
                    self.setEnabled(False)  # 禁用主窗口
                    self.detect_image(path=self.source,show=False,save_to_file=True,project=folder_path)


        else:
            error_message = "请先选择文件数据。"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
    #显示进度条
    # def show_progress(self):
    #     SI.save_imgs_win=Detect_saveImgs_win()
    #     SI.save_imgs_win.show()
    #     # SI.save_imgs_win.exec_()
    # def updata_progress(self,values):
    #     SI.save_imgs_win.progressBar.setValue(int((values[0] / values[1]) * 100))
    #获取图片文件信息
    def get_file_information(self,path):
        Images_num=os.listdir(path)
        self.TotalNum=len(Images_num)
        # print(self.TotalNum)
        if self.TotalNum>0:
            self.label_image_number.setText(str(self.TotalNum))#显示图片总数
    #选择数据文件夹事件
    def pushButton_dirIMage_click_my(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择数据文件夹", './')
        if folder_path == None or not os.path.exists(folder_path):
            error_message = "没有选择文件。"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
        else:
            self.source = folder_path
            #读取文件信息
            self.get_file_information(str(self.source))
            print("获取文件路径成功: " + str(self.source))
            # print(self.source)
            if folder_path:
                self.image_paths = []
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                        file_path = os.path.join(folder_path, file_name)
                        self.image_paths.append(file_path)

                if self.image_paths:
                    self.current_image_index = 0
                    self.label_image_indxe.setText(str(self.current_image_index))#显示索引
                    image_path = self.image_paths[self.current_image_index]
                    self.image_path = image_path
                    pixmap = QPixmap(image_path)
                    self.label_13.setPixmap(pixmap)#显示图片

    #更改置信度
    def pushButton_confidence_click_my (self):
        self.conf_thres=self.doubleSpinBox_confidence.value()

        image_path=self.image_path
        self.label_image_dir.setText(str(image_path))
        self.label_image_indxe.setText(str(self.current_image_index + 1))
        pixmap = QPixmap(image_path).scaled(self.label_13.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label_13.setPixmap(pixmap)
        self.detect_image(path=image_path)

    #切换图按钮
    def switch_next_image(self):
        if self.image_paths:
            self.current_image_index += 1
            if self.current_image_index >= len(self.image_paths):
                self.current_image_index = 0

            image_path = self.image_paths[self.current_image_index]
            if os.path.exists(image_path):
                self.image_path=image_path
                self.label_image_dir.setText(str(image_path))
                self.label_image_indxe.setText(str(self.current_image_index+1))
                pixmap = QPixmap(image_path).scaled(self.label_13.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
                self.label_13.setPixmap(pixmap)
                self.detect_image(path=image_path)

    def switch_prev_image(self):
        if self.image_paths:
            self.current_image_index -= 1
            if self.current_image_index < 0:
                self.current_image_index = len(self.image_paths) - 1
            image_path = self.image_paths[self.current_image_index]
            if os.path.exists(image_path):
                self.image_path = image_path
                self.label_image_dir.setText(str(image_path))
                self.label_image_indxe.setText(str(self.current_image_index + 1))
                pixmap = QPixmap(image_path).scaled(self.label_13.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
                self.label_13.setPixmap(pixmap)
                self.detect_image(path=image_path)

    # 选择单个图片事件
    def pushButton_oneImage_click_my(self):
        if self.model!=None:
            fname1 , _= QFileDialog.getOpenFileName(self, "选择图片", "./", "Image (*.tiff *.jpeg *.png)")
            if fname1 == None or not os.path.exists(fname1):
                error_message = "没有选择文件。"
                QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
            else:
                self.source = fname1
                self.image_path=fname1
                print("获取文件路径成功")
                print(self.source)
                self.index_img = 1
                self.label_image_indxe.setText(str(self.index_img))
                self.label_image_dir.setText(str(self.source))
                self.label_image_number.setText(str(0))
                pixmap = QPixmap(fname1)
                self.label_13.setPixmap(pixmap)  # 显示图片
                self.detect_image(path=self.source)
        else:
            error_message = "请先选择模型"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)

    def sttask_finishedop(self,value):
        self.setEnabled(True)  # 启用主窗口

    #单线程检测图片
    def detect_image(self,path,show=True,save_to_file=False,project=None):
            if self.model!=None:
                if (os.path.exists(path)):
                    det_Image_thread=DetThread()#检测图片
                    det_Image_thread.model = self.model
                    det_Image_thread.source=path
                    det_Image_thread.save_to_file = save_to_file
                    det_Image_thread.project = project
                    det_Image_thread.conf_thres=self.conf_thres
                    det_Image_thread.finished.connect(self.sttask_finishedop)
                    # det_I.signal_progress_update.connect(self.updata_progress)#更新进度条信息
                    det_Image_thread.is_show = show
                    if show:
                        det_Image_thread.is_show=show
                        det_Image_thread.send_img.connect(lambda x: self.show_image(x, self.label))
                        det_Image_thread.send_msg.connect(lambda x: self.set_time(x))
                    # self.det_Image_thread.setDaemon(True)  # 将子线程设置为守护线程
                    det_Image_thread.start()
                else:
                    error_message = "图片不存在。"
                    QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)

            else:
                error_message = "没有导入模型。"
                QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
    #设置显示时间
    def set_time(self,time):
        self.label_time.setText(time)

    def eventFilter(self, obj, event):
        if obj is self.label_13 and event.type() == QEvent.Resize:
            self.scale_image(self.label_13)
        if obj is self.label and event.type() == QEvent.Resize:
            self.scale_image(self.label)
        return super().eventFilter(obj, event)

    #缩放图像
    def scale_image(self,label_112121113):
        pixmap = label_112121113.pixmap()
        if pixmap is None:
            return
        scaled_pixmap = pixmap.scaled(label_112121113.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        label_112121113.setPixmap(scaled_pixmap)

    #选择权重事件
    def pushButton_weight_click_my(self):
        fname1 , _= QFileDialog.getOpenFileName(self, "选择.pt", "./datasets/detect", "weight documents (*.pt)")
        if fname1 == None or not os.path.exists(fname1):
            error_message = "没有选择文件。"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
        else:
            self.weights = fname1
            if os.path.exists(fname1):
                  self.get_models()#得到模型
            print("获取文件路径成功")
            print(self.weights)

    #获取模型文件
    def get_models(self):
        g_m1=get_model()
        g_m1.weights=self.weights
        g_m1.data=self.data
        g_m1.signal_done.connect(lambda x:self.getmodel(x,g_m1))
        g_m1.start()

    def getmodel(self,val,g_m1):
        self.model=g_m1.model

    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self, "Xpath Robot", "是否要退出，退出后未保存的数据将被清除?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if (result == QtWidgets.QMessageBox.Yes):
            event.accept()
        else:
            event.ignore()

    #保存当前图片
    def pushButton_saveImage_click_my(self):
        if self.source!=None and os.path.exists(self.source):
            self.make_dir()
            im0=self.imageo
            filename=os.path.basename(self.image_path)
            save_path='./runs/detect/'+self.projectName+'/'+filename
            cv2.imwrite(save_path, im0)
        else:
            error_message = "图片不存在"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)

    # @staticmethod
    #显示图片
    def show_image(self,image, label):
        try:


            self.imageo=image#保存图片
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # 将QImage对象转换为QPixmap对象，并设置为QLabel的图片
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            label.setPixmap(scaled_pixmap)
            #label.setScaledContents(True)
            #self.scale_image(label)
            # height, width, channel = image.shape
            # bytes_per_line = 3 * width
            # q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            #
            # # 将QImage对象转换为QPixmap对象，并设置为QLabel的图片
            # pixmap = QPixmap.fromImage(q_image)
            # label.setPixmap(pixmap)

        except Exception as e:
            print(repr(e))


    #创建文件夹操作
    def make_dir(self):
        # print(self.projectName)
        dir='./runs/detect/'+self.projectName

        if not os.path.exists(dir):
            os.makedirs(dir)
            print("目录已创建")
        # else:
            # print("目录已存在")


#新建测试工程界面
class Detect_new_win(QDialog,detect_new_window):
    def __init__(self,parent=None):
        super(Detect_new_win,self).__init__(parent)
        self.setupUi(self)
        self.weightDir=None
        self.classeseDir=None
        self.projectName=None
        self.groupBox.setStyleSheet("QGroupBox {border: 0px solid transparent;}")
        self.pushButton_weight.clicked.connect(self.pushButton_weight_click_my)
        self.pushButton_classese.clicked.connect(self.pushButton_classese_click_my)
        self.pushButton_ok.clicked.connect(self.pushButton_ok_click_my)
        self.pushButton_cancle.clicked.connect(self.close)
        # 按钮事件
    def pushButton_weight_click_my(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        fname1,_ = file_dialog.getOpenFileName(None, "选择文件",  "./", "weight documents (*.pt)",options=options)
        if not os.path.exists(fname1) or fname1==None :
            # 文件路径为空，显示错误提示
            error_message = "没有选择文件。"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
        else:
            self.weightDir=fname1
            print("获取模型文件路径成功")
            print(self.weightDir)
    # 按钮事件
    def pushButton_classese_click_my(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        fname1, _ = file_dialog.getOpenFileName(None, "选择文件", "", "classese documents (*.txt)", options=options)
        if not os.path.exists(fname1) or fname1==None :
            # 文件路径为空，显示错误提示
            error_message = "没有选择文件。"
            QMessageBox.critical(None, "错误", error_message, QMessageBox.Ok)
        else:
            self.classeseDir = fname1
            print("获取classea文件路径成功")
            print(self.classeseDir)

    #定义按钮OK
    def pushButton_ok_click_my(self):
        self.projectName = self.lineEdit_projectname.text()
        if self.projectName !='' and os.path.exists(self.classeseDir) and os.path.exists(self.weightDir):
            self.projectName = self.lineEdit_projectname.text()
            self.get_diretory_all()#创建文件夹
            file_class_path=self.classeseDir
            self.classes = get_classes(file_class_path)
            self.res = { "names": self.classes}
            self.projectDiretory = (os.path.dirname(__file__) +r"\datasets\detect\{0}".format(self.projectName))
            print(self.projectDiretory)
            path_classese = (self.projectDiretory + '\{0}.yaml'.format(self.projectName))
            print(path_classese)
            self.write_yaml(input=path_classese,res=self.res)
            source_file=self.weightDir
            destination_folder=self.projectDiretory
            shutil.copy(source_file, destination_folder)
            self.close()
        else:
            mess_box = QMessageBox(QMessageBox.Warning, '提示', '输入数据无效')
            mess_box.exec_()

    def write_yaml(self,input,res):
        with open(input, "w", encoding="utf-8") as f:
            yaml.dump(res, f,allow_unicode=True)
        f.close()

    def get_diretory_all(self):
        if not os.path.exists('./datasets'):
            os.makedirs('./datasets')
        if not os.path.exists('./datasets/detect'):
            os.makedirs('./datasets/detect')
        if not os.path.exists('./datasets/detect/{0}'.format(self.projectName)):
            os.makedirs('./datasets/detect/{0}'.format(self.projectName))

# class Detect_saveImgs_win(QDialog,save_imgs_Win):
#     def __init__(self,parent=None):
#         super(Detect_saveImgs_win,self).__init__(parent)
#         self.setupUi(self)
#
#         self.progressBar.setValue(0)
#         self.progressBar.setRange(0, 100)
#     def updata_progress(self,values):
#         SI.save_imgs_win.progressBar.setValue(int((values[0] / values[1]) * 100))

# 导入测试界面
class Detect_import_win(QDialog,detect_import_window):
    def __init__(self,parent=None):
        super(Detect_import_win,self).__init__(parent)
        self.setupUi(self)
        self.groupBox.setStyleSheet("QGroupBox {border: 0px solid transparent;}")
        self.weightDir = None
        self.projectName = None
        self.cfgDir = None
        self.file = './datasets/detect'  # 当前目录
        self.ReadData()
        self.pushButton_ok.clicked.connect(self.pushButton_ok_click_my)
        self.pushButton_cancel.clicked.connect(self.close)
    def pushButton_ok_click_my(self):
        self.projectName = self.comboBox_import.currentText()
        GLV.set_value_1('detect_ProjectFilename', self.projectName)
        if self.projectName != '':
            projectDiretory = (os.path.dirname(__file__) + r'\datasets\detect\%s' % (self.projectName))
            file_my=os.listdir(projectDiretory)
            for subdir in file_my:
                if subdir.endswith('.pt'):
                    self.weightDir=projectDiretory+r'\{}'.format(subdir)
                    GLV.set_value_1('detect_weight',self.weightDir)
                if subdir.endswith('.yaml'):
                    self.cfgDir=projectDiretory+r'\{}'.format(subdir)
                    GLV.set_value_1('detect_data', self.cfgDir)
                break
            self.close()
            self.show_de_ui()
        else:
            mess_box = QMessageBox(QMessageBox.Warning, '提示', '输入数据无效')
            mess_box.exec_()
    def ReadData(self):
        if os.path.exists(self.file):
            self.ageFile=os.listdir(self.file)
            self.comboBox_import.addItems(self.ageFile)
        else:
            pass
    def show_de_ui(self):
        SI.detect_ui = Detect_window()
        SI.detect_ui.show()

 #删除测试工程
class Detect_delete_win(QDialog,detect_delete_window):
    def __init__(self,parent=None):
        super(Detect_delete_win,self).__init__(parent)
        self.setupUi(self)
        self.file = './datasets/detect'
        self.ReadData()
        self.groupBox.setStyleSheet("QGroupBox {border: 0px solid transparent;}")
        self.pushButton_ok.clicked.connect(self.pushButton_ok_my)

    def pushButton_ok_my(self):
        if os.path.exists(self.file) :
            self.ageFile_final=self.comboBox_delect.currentText()
            # print(os.path.dirname(__file__))
            self.agediretory=(os.path.dirname(__file__)+r'\datasets\detect\%s' % (self.ageFile_final))
            # print(self.agediretory)
            if os.path.exists(self.agediretory):
                result = QtWidgets.QMessageBox.question(self, "确认", "确认删除工程？",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if (result == QtWidgets.QMessageBox.Yes):  # 按下“yes”触发
                    self.thread = threading.Thread(target=self.Delete_data(dir=self.agediretory))
                    self.thread.start()
                    self.thread.join()
                    if not os.path.exists(self.agediretory):
                        self.comboBox_delect.clear()
                        self.ReadData()
                        error_message = "删除完成。"
                        QMessageBox.information(None, "提示", error_message, QMessageBox.Ok)

                else:
                    return

            else:
                mess_box2 = QMessageBox(QMessageBox.Warning, '警告', '不存在如此文件夹')
                mess_box2.exec_()
    def ReadData(self):
        if os.path.exists(self.file):
            self.ageFile=os.listdir(self.file)
            self.comboBox_delect.addItems(self.ageFile)
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
        self.openTensor=True #
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
        self.spinBox_batch_size.setRange(2, 200)
        self.spinBox_batch_size.setValue(16)  # 设置当前值；
        self.spinBox_batch_size.setSingleStep(2)
        self.spinBox_worker.setRange(0, 40)
        self.spinBox_worker.setValue(8)  # 设置当前值；
        self.spinBox_worker.setSingleStep(1)
        self.spinBox_patience.setRange(50, 2000)
        self.spinBox_patience.setSingleStep(10)
        self.spinBox_patience.setValue(100)  # 设置当前值；
        self.spinBox_epoch.setRange(1, 214748364)
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
            self.batchsize=8
            self.spinBox_batch_size.setValue(self.batchsize)
            self.workers=3
            self.spinBox_worker.setValue(self.workers)
            self.epoches=300
            self.spinBox_epoch.setValue(self.epoches)
            self.patience=100
            self.spinBox_patience.setValue(self.patience)
            self.optimizer="SGD"
            self.QComboBox_optimize.setCurrentText(self.optimizer)
        if suggest_item=="配置2":
            self.batchsize = 16
            self.spinBox_batch_size.setValue(self.batchsize)
            self.workers = 6
            self.spinBox_worker.setValue(self.workers)
            self.epoches = 400
            self.spinBox_epoch.setValue(self.epoches)
            self.patience = 100
            self.spinBox_patience.setValue(self.patience)
            self.optimizer = "SGD"
            self.QComboBox_optimize.setCurrentText(self.optimizer)
        if suggest_item=="配置3":
            self.batchsize=16
            self.spinBox_batch_size.setValue(self.batchsize)
            self.workers=8
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
                stop_train='False_stop'
                GLV.set_value_1('stop_train', stop_train)
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
        if self.openTensor:
            T_tensor = threading.Thread(target=self.open_tensorboard)  #
            T_tensor.start()
            self.openTensor=False
        # self.TensorboardWIN = Tensorbpard_Window()
        # self.TensorboardWIN.show()
        # subprocess.Popen(["google-chrome", "http://localhost:6006"])  # 用你的浏览器和TensorBoard URL 替代
        tensorboard_url = "http://localhost:6006"  # 示例URL，请替换为你的TensorBoard地址
        # 使用默认浏览器打开TensorBoard页面
        webbrowser.open(tensorboard_url)
    def open_tensorboard(self):
        tensorboard_path = os.path.dirname(os.path.realpath(sys.argv[0])) + '/tensorboard.exe'  # exe文件绝对路径
        # print(tensorboard_path)
        sys.path.append(tensorboard_path)  # exe加入环境变量
        open_tensorboard_command = tensorboard_path + ' --logdir runs/train'  # cmd命令行
        # print(open_tensorboard_command)
        os.system(open_tensorboard_command)  # 利用cmd激活
        # os.system('tensorboard --logdir runs/train')

    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self, "Xpath Robot", "是否要退出，退出后未保存的数据将被清除?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if (result == QtWidgets.QMessageBox.Yes):
            stop_train = 'True_stop'  # 停止训练变量
            GLV.set_value_1('stop_train', stop_train)  # 赋值，在train.py 404行读取,强行退出时依旧会训练完当前epoch并保存结果
            event.accept()
        else:
            event.ignore()

#样本统计界面
# class label_sum_ui(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.labelUI()
#     def labelUI(self):
#         self.resize(400, 400)
#         label_sum = GLV.get_value('label_sum')
#         self.table_widget = QTableWidget(len(label_sum), 2)  # 创建一个2列的表格
#         self.table_widget.setHorizontalHeaderLabels(['类名', '数量'])
#         self.label_xml = QTabWidget(self) #创建一个用于收纳表格的控件
#         self.label_xml.setGeometry(0, 0, 300, 400)
#         #表格导出按钮
#         self.button_export = QPushButton('导出为xml文件',self)
#         self.button_export.setGeometry(301,10,99,30)
#         self.button_export.clicked.connect(self.export_to_xml)
#     def label_sum_show(self):
#         label_sum = GLV.get_value('label_sum')
#         print(label_sum)
#         # 清空表格内容
#         self.table_widget.clearContents()
#         # 将字典的内容添加到表格中
#         for row, (key, value) in enumerate(label_sum.items()):
#             key_item = QTableWidgetItem(key)
#             value_item = QTableWidgetItem(str(value))
#             self.table_widget.setItem(row, 0, key_item)
#             self.table_widget.setItem(row, 1, value_item)
#         self.label_xml.addTab(self.table_widget, '样本统计')
#     import xml.etree.ElementTree as ET
#     def export_to_xml(self):
#         root = ET.Element("table_data")
#         filename = './data/datasets/labels.xml'
#         for row in range(self.table_widget.rowCount()):
#             row_element = ET.SubElement(root, "row")
#             for column in range(self.table_widget.columnCount()):
#                 cell_data = self.table_widget.item(row, column).text()
#                 cell_element = ET.SubElement(row_element, f"column_{column}")
#                 cell_element.text = cell_data
#
#         tree = ET.ElementTree(root)
#         tree.write(filename, encoding="utf-8", xml_declaration=True)
#         QMessageBox.information(self,'tips','导出成功')

class Tensorbpard_Window(QMainWindow,tensorboard_MainWindow):
    def __init__(self,parent=None):
        super(Tensorbpard_Window,self).__init__(parent)
        self.setupUi(self)
        self.webEngineView = QtWebEngineWidgets.QWebEngineView(self.frame)
        url = "http://localhost:6006/"
        self.webEngineView.load(QUrl(url))
        # print(self.frame.width(), self.frame.height())
        self.webEngineView.setGeometry(0, 0, 1000, 800)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_web_view_size()

    def adjust_web_view_size(self):
        self.webEngineView.setFixedSize(self.centralwidget.size())

    def closeEvent(self, event):
        # 销毁视图对象
        self.webEngineView.deleteLater()
        event.accept()

    # def __del__(self):
        # print("调用__del__() 销毁对象，释放其空间")

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
                        self.QComboBox_file.clear()
                        self.ReadData()
                        error_message = "删除完成。"
                        QMessageBox.information(None, "提示", error_message, QMessageBox.Ok)
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
    multiprocessing.freeze_support()
    app = QApplication([])#实例化
    mainWin =MainWindow()
    mainWin.show()
    sys.exit(app.exec())
    # app.exec_()


