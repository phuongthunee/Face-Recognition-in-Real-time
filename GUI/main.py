# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_FaceRecognition(object):
    def setupUi(self, FaceRecognition):
        FaceRecognition.setObjectName("FaceRecognition")
        FaceRecognition.setWindowModality(QtCore.Qt.NonModal)
        FaceRecognition.setEnabled(True)
        FaceRecognition.resize(532, 377)
        self.horizontalLayout = QtWidgets.QHBoxLayout(FaceRecognition)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.hzTimeLayout = QtWidgets.QHBoxLayout()
        self.hzTimeLayout.setObjectName("hzTimeLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.date = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        self.date.setFont(font)
        self.date.setObjectName("date")
        self.horizontalLayout_2.addWidget(self.date)
        self.dateLabel = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        self.dateLabel.setFont(font)
        self.dateLabel.setText("")
        self.dateLabel.setObjectName("dateLabel")
        self.horizontalLayout_2.addWidget(self.dateLabel)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.time = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        self.time.setFont(font)
        self.time.setObjectName("time")
        self.horizontalLayout_3.addWidget(self.time)
        self.timeLabel = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        self.timeLabel.setFont(font)
        self.timeLabel.setText("")
        self.timeLabel.setObjectName("timeLabel")
        self.horizontalLayout_3.addWidget(self.timeLabel)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.hzTimeLayout.addLayout(self.gridLayout_2)
        self.gridLayout.addLayout(self.hzTimeLayout, 0, 0, 1, 1)
        self.image_label = QtWidgets.QLabel(FaceRecognition)
        self.image_label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("image_label")
        self.gridLayout.addWidget(self.image_label, 1, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi(FaceRecognition)
        QtCore.QMetaObject.connectSlotsByName(FaceRecognition)

    def retranslateUi(self, FaceRecognition):
        _translate = QtCore.QCoreApplication.translate
        FaceRecognition.setWindowTitle(_translate("FaceRecognition", "Facial Recognition Attendance System"))
        self.date.setText(_translate("FaceRecognition", "Date:"))
        self.time.setText(_translate("FaceRecognition", "Time:"))
        self.image_label.setText(_translate("FaceRecognition", "Camera"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FaceRecognition = QtWidgets.QWidget()
    ui = Ui_FaceRecognition()
    ui.setupUi(FaceRecognition)
    FaceRecognition.show()
    sys.exit(app.exec_())

