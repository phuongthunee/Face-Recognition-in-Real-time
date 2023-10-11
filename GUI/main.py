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
        self.mainLayout = QtWidgets.QVBoxLayout(FaceRecognition)
        self.mainLayout.setObjectName("mainLayout")
        self.hzTimeLayout = QtWidgets.QHBoxLayout()
        self.hzTimeLayout.setObjectName("hzTimeLayout")
        self.date = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.date.setFont(font)
        self.date.setObjectName("date")
        self.hzTimeLayout.addWidget(self.date)
        self.dateLabel = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.dateLabel.setFont(font)
        self.dateLabel.setText("")
        self.dateLabel.setObjectName("dateLabel")
        self.hzTimeLayout.addWidget(self.dateLabel)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.hzTimeLayout.addItem(spacerItem)
        self.time = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.time.setFont(font)
        self.time.setObjectName("time")
        self.hzTimeLayout.addWidget(self.time)
        self.timeLabel = QtWidgets.QLabel(FaceRecognition)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.timeLabel.setFont(font)
        self.timeLabel.setText("")
        self.timeLabel.setObjectName("timeLabel")
        self.hzTimeLayout.addWidget(self.timeLabel)
        self.mainLayout.addLayout(self.hzTimeLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.mainLayout.addItem(spacerItem1)
        self.image_label = QtWidgets.QLabel(FaceRecognition)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("image_label")
        self.mainLayout.addWidget(self.image_label)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.mainLayout.addItem(spacerItem2)
        self.hboxlayout = QtWidgets.QHBoxLayout()
        self.hboxlayout.setObjectName("hboxlayout")
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.hboxlayout.addItem(spacerItem3)
        self.view = QtWidgets.QPushButton(FaceRecognition)
        self.view.setObjectName("view")
        self.hboxlayout.addWidget(self.view)
        spacerItem4 = QtWidgets.QSpacerItem(150, 50, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.hboxlayout.addItem(spacerItem4)
        self.mainLayout.addLayout(self.hboxlayout)

        self.retranslateUi(FaceRecognition)
        QtCore.QMetaObject.connectSlotsByName(FaceRecognition)

    def retranslateUi(self, FaceRecognition):
        _translate = QtCore.QCoreApplication.translate
        FaceRecognition.setWindowTitle(_translate("FaceRecognition", "Facial Recognition Attendance System"))
        FaceRecognition.setStyleSheet(_translate("FaceRecognition", "\n"
"    QWidget {\n"
"            background-color: #ebffeb;  \n"
"        }\n"
"        QLabel {\n"
"            color: #333333;\n"
"        }\n"
"        QLabel#image_label {\n"
"            border: 3px solid #006400;  \n"
"            background-color: #ebffeb;  \n"
"        }\n"
"        QPushButton {\n"
"            background-color: #006400;\n"
"            color: white;\n"
"            border-radius: 10px;  \n"
"            padding: 15px 30px;  \n"
"            font-size: 18px;  \n"
"            font-weight: bold;  \n"
"        }\n"
"        QPushButton:hover {\n"
"            background-color: #006400;\n"
"        }\n"
"   "))
        self.date.setText(_translate("FaceRecognition", "Date:"))
        self.time.setText(_translate("FaceRecognition", "Time:"))
        self.image_label.setText(_translate("FaceRecognition", "Camera"))
        self.view.setText(_translate("FaceRecognition", "View Attendance"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FaceRecognition = QtWidgets.QWidget()
    ui = Ui_FaceRecognition()
    ui.setupUi(FaceRecognition)
    FaceRecognition.show()
    sys.exit(app.exec_())

