
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'import_data_Window.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets

class import_MainWindow(object):
        def setupUi(self, Dialog):
            Dialog.setObjectName("Dialog")
            Dialog.resize(700, 500)
            Dialog.setMinimumSize(QtCore.QSize(700, 500))
            Dialog.setMaximumSize(QtCore.QSize(700, 500))
            self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
            self.verticalLayout.setObjectName("verticalLayout")
            spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.verticalLayout.addItem(spacerItem)
            self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
            self.horizontalLayout_3.setObjectName("horizontalLayout_3")
            spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.horizontalLayout_3.addItem(spacerItem1)
            self.label = QtWidgets.QLabel(Dialog)
            self.label.setStyleSheet("font: 14pt \"新宋体\";")
            self.label.setObjectName("label")
            self.horizontalLayout_3.addWidget(self.label)
            spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.horizontalLayout_3.addItem(spacerItem2)
            self.verticalLayout.addLayout(self.horizontalLayout_3)
            spacerItem3 = QtWidgets.QSpacerItem(20, 27, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.verticalLayout.addItem(spacerItem3)
            self.horizontalLayout = QtWidgets.QHBoxLayout()
            self.horizontalLayout.setObjectName("horizontalLayout")
            spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.horizontalLayout.addItem(spacerItem4)
            self.QComboBox_file = QtWidgets.QComboBox(Dialog)
            self.QComboBox_file.setMinimumSize(QtCore.QSize(300, 30))
            self.QComboBox_file.setMaximumSize(QtCore.QSize(300, 30))
            self.QComboBox_file.setStyleSheet("font: 14pt \"新宋体\";")
            self.QComboBox_file.setObjectName("QComboBox_file")
            self.QComboBox_file.addItem("")
            self.horizontalLayout.addWidget(self.QComboBox_file)
            spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.horizontalLayout.addItem(spacerItem5)
            self.verticalLayout.addLayout(self.horizontalLayout)
            spacerItem6 = QtWidgets.QSpacerItem(20, 115, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.verticalLayout.addItem(spacerItem6)
            self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
            self.horizontalLayout_2.setObjectName("horizontalLayout_2")
            spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.horizontalLayout_2.addItem(spacerItem7)
            self.pushButton_ok = QtWidgets.QPushButton(Dialog)
            self.pushButton_ok.setObjectName("pushButton_ok")
            self.horizontalLayout_2.addWidget(self.pushButton_ok)
            self.pushButton_cancel = QtWidgets.QPushButton(Dialog)
            self.pushButton_cancel.setObjectName("pushButton_cancel")
            self.horizontalLayout_2.addWidget(self.pushButton_cancel)
            self.horizontalLayout_2.setStretch(0, 2)
            self.horizontalLayout_2.setStretch(1, 1)
            self.horizontalLayout_2.setStretch(2, 1)
            self.verticalLayout.addLayout(self.horizontalLayout_2)
            self.verticalLayout.setStretch(0, 2)
            self.verticalLayout.setStretch(1, 4)
            self.verticalLayout.setStretch(2, 1)
            self.verticalLayout.setStretch(3, 2)
            self.verticalLayout.setStretch(4, 4)
            self.verticalLayout.setStretch(5, 2)

            self.retranslateUi(Dialog)
            QtCore.QMetaObject.connectSlotsByName(Dialog)

        def retranslateUi(self, Dialog):
            _translate = QtCore.QCoreApplication.translate
            Dialog.setWindowTitle(_translate("Dialog", "导入训练工程"))
            self.label.setText(_translate("Dialog", "请导入数据（已有训练工程）"))
            self.QComboBox_file.setItemText(0, _translate("Dialog", "选择导入文件夹"))
            self.pushButton_ok.setText(_translate("Dialog", "确定"))
            self.pushButton_cancel.setText(_translate("Dialog", "取消"))
