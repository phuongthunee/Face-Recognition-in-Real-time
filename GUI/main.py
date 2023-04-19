# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from GUI.information import Ui_Information

class Ui_FaceRecognition(object):
    def changeWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Information()
        self.ui.setupUi(self.window)
        self.window.show()

    
    def setupUi(self, FaceRecognition):
        FaceRecognition.setObjectName("FaceRecognition")
        FaceRecognition.resize(1000, 452)
        FaceRecognition.setMinimumSize(QtCore.QSize(1000, 400))
        self.centralwidget = QtWidgets.QWidget(FaceRecognition)
        self.centralwidget.setObjectName("centralwidget")
        self.videoLabel = QtWidgets.QLabel(self.centralwidget)
        self.videoLabel.setGeometry(QtCore.QRect(20, 10, 500, 281))
        self.videoLabel.setMinimumSize(QtCore.QSize(500, 150))
        self.videoLabel.setObjectName("videoLabel")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(150, 310, 241, 81))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.clock = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.clock.setContentsMargins(0, 0, 0, 0)
        self.clock.setVerticalSpacing(0)
        self.clock.setObjectName("clock")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.date = QtWidgets.QLabel(self.gridLayoutWidget)
        self.date.setObjectName("date")
        self.horizontalLayout_2.addWidget(self.date)
        self.dateLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.dateLabel.setText("")
        self.dateLabel.setObjectName("dateLabel")
        self.horizontalLayout_2.addWidget(self.dateLabel)
        self.clock.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.time = QtWidgets.QLabel(self.gridLayoutWidget)
        self.time.setObjectName("time")
        self.horizontalLayout_3.addWidget(self.time)
        self.timeLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.timeLabel.setText("")
        self.timeLabel.setObjectName("timeLabel")
        self.horizontalLayout_3.addWidget(self.timeLabel)
        self.clock.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        FaceRecognition.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(FaceRecognition)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 18))
        self.menubar.setObjectName("menubar")
        FaceRecognition.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(FaceRecognition)
        self.statusbar.setObjectName("statusbar")
        FaceRecognition.setStatusBar(self.statusbar)

        self.retranslateUi(FaceRecognition)
        QtCore.QMetaObject.connectSlotsByName(FaceRecognition)

    def retranslateUi(self, FaceRecognition):
        _translate = QtCore.QCoreApplication.translate
        FaceRecognition.setWindowTitle(_translate("FaceRecognition", "Face Recognition"))
        self.videoLabel.setText(_translate("FaceRecognition", "webcam"))
        self.date.setText(_translate("FaceRecognition", "Date:"))
        self.time.setText(_translate("FaceRecognition", "Time:"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FaceRecognition = QtWidgets.QMainWindow()
    ui = Ui_FaceRecognition()
    ui.setupUi(FaceRecognition)
    FaceRecognition.show()
    sys.exit(app.exec_())