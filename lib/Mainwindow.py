# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 500)
        MainWindow.setMinimumSize(QtCore.QSize(700, 500))
        MainWindow.setMaximumSize(QtCore.QSize(1200, 900))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(14)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 700, 29))
        self.menubar.setObjectName("menubar")
        self.btn_train = QtWidgets.QMenu(self.menubar)
        self.btn_train.setBaseSize(QtCore.QSize(20, 20))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(14)
        self.btn_train.setFont(font)
        self.btn_train.setObjectName("btn_train")
        self.btn_detect = QtWidgets.QMenu(self.menubar)
        self.btn_detect.setObjectName("btn_detect")
        self.btn_labeling = QtWidgets.QMenu(self.menubar)
        self.btn_labeling.setObjectName("btn_labeling")
        self.menuff = QtWidgets.QMenu(self.menubar)
        self.menuff.setObjectName("menuff")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_new = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.action_new.setFont(font)
        self.action_new.setObjectName("action_new")
        self.action_import = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.action_import.setFont(font)
        self.action_import.setObjectName("action_import")
        self.action_delete = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.action_delete.setFont(font)
        self.action_delete.setObjectName("action_delete")
        self.action1_new = QtWidgets.QAction(MainWindow)
        self.action1_new.setObjectName("action1_new")
        self.action2_import = QtWidgets.QAction(MainWindow)
        self.action2_import.setObjectName("action2_import")
        self.action_delete_2 = QtWidgets.QAction(MainWindow)
        self.action_delete_2.setObjectName("action_delete_2")
        self.action4_conbime = QtWidgets.QAction(MainWindow)
        self.action4_conbime.setObjectName("action4_conbime")
        self.actionsss = QtWidgets.QAction(MainWindow)
        self.actionsss.setObjectName("actionsss")
        self.action_help = QtWidgets.QAction(MainWindow)
        self.action_help.setObjectName("action_help")
        self.action13344 = QtWidgets.QAction(MainWindow)
        self.action13344.setObjectName("action13344")
        self.btn_train.addAction(self.action_new)
        self.btn_train.addAction(self.action_import)
        self.btn_train.addAction(self.action_delete)
        self.btn_detect.addAction(self.action1_new)
        self.btn_detect.addAction(self.action2_import)
        self.btn_detect.addAction(self.action_delete_2)
        self.btn_labeling.addAction(self.actionsss)
        self.menuff.addAction(self.action_help)
        self.menuff.addAction(self.action13344)
        self.menubar.addAction(self.btn_train.menuAction())
        self.menubar.addAction(self.btn_detect.menuAction())
        self.menubar.addAction(self.btn_labeling.menuAction())
        self.menubar.addAction(self.menuff.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DeepLearning Model---Yolov5"))
        self.btn_train.setTitle(_translate("MainWindow", "训练模型"))
        self.btn_detect.setTitle(_translate("MainWindow", "检测数据"))
        self.btn_labeling.setTitle(_translate("MainWindow", "数据标注"))
        self.menuff.setTitle(_translate("MainWindow", "其他"))
        self.action_new.setText(_translate("MainWindow", "新建训练工程"))
        self.action_import.setText(_translate("MainWindow", "导入已有工程"))
        self.action_delete.setText(_translate("MainWindow", "删除训练工程"))
        self.action1_new.setText(_translate("MainWindow", "新建测试工程"))
        self.action2_import.setText(_translate("MainWindow", "导入测试工程"))
        self.action_delete_2.setText(_translate("MainWindow", "删除测试工程"))
        self.action4_conbime.setText(_translate("MainWindow", "联合测试"))
        self.actionsss.setText(_translate("MainWindow", "开始标注"))
        self.action_help.setText(_translate("MainWindow", "帮助"))
        self.action13344.setText(_translate("MainWindow", "退出"))
