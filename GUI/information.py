# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'information.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Information(object):
    def setupUi(self, Information):
        Information.setObjectName("Information")
        Information.setWindowModality(QtCore.Qt.NonModal)
        Information.setEnabled(True)
        Information.resize(532, 291)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Information)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imageLabel = QtWidgets.QLabel(Information)
        self.imageLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        self.imageLabel.setFont(font)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setObjectName("imageLabel")
        self.verticalLayout.addWidget(self.imageLabel)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.nameLabel = QtWidgets.QLabel(Information)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.nameLabel.setFont(font)
        self.nameLabel.setText("")
        self.nameLabel.setObjectName("nameLabel")
        self.verticalLayout_2.addWidget(self.nameLabel)
        self.timeInLabel = QtWidgets.QLabel(Information)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.timeInLabel.setFont(font)
        self.timeInLabel.setText("")
        self.timeInLabel.setObjectName("timeInLabel")
        self.verticalLayout_2.addWidget(self.timeInLabel)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.return_bt = QtWidgets.QPushButton(Information)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.return_bt.setFont(font)
        self.return_bt.setObjectName("return_bt")
        self.horizontalLayout_2.addWidget(self.return_bt)
        self.close_bt = QtWidgets.QPushButton(Information)
        font = QtGui.QFont()
        font.setFamily("STXinwei")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.close_bt.setFont(font)
        self.close_bt.setObjectName("close_bt")
        self.horizontalLayout_2.addWidget(self.close_bt)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(Information)
        QtCore.QMetaObject.connectSlotsByName(Information)

    def retranslateUi(self, Information):
        _translate = QtCore.QCoreApplication.translate
        Information.setWindowTitle(_translate("Information", "Noti"))
        self.imageLabel.setText(_translate("Information", "Camera"))
        self.return_bt.setText(_translate("Information", "Return"))
        self.close_bt.setText(_translate("Information", "Close"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Information = QtWidgets.QWidget()
    ui = Ui_Information()
    ui.setupUi(Information)
    Information.show()
    sys.exit(app.exec_())

