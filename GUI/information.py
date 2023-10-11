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
        Information.resize(921, 389)
        Information.setStyleSheet("QWidget { background-color: #e0e0e0; }")
        self.horizontalLayout = QtWidgets.QHBoxLayout(Information)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.imageLabel = QtWidgets.QLabel(Information)
        self.imageLabel.setMinimumSize(QtCore.QSize(532, 377))
        self.imageLabel.setMaximumSize(QtCore.QSize(532, 377))
        self.imageLabel.setObjectName("imageLabel")
        self.horizontalLayout.addWidget(self.imageLabel)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.nameLabel = QtWidgets.QLabel(Information)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.verticalLayout_2.addWidget(self.nameLabel)
        self.timeInLabel = QtWidgets.QLabel(Information)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.timeInLabel.setFont(font)
        self.timeInLabel.setObjectName("timeInLabel")
        self.verticalLayout_2.addWidget(self.timeInLabel)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.check_bt = QtWidgets.QPushButton(Information)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        self.check_bt.setFont(font)
        self.check_bt.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; }")
        self.check_bt.setObjectName("check_bt")
        self.horizontalLayout_2.addWidget(self.check_bt)
        self.checkout_bt = QtWidgets.QPushButton(Information)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        self.checkout_bt.setFont(font)
        self.checkout_bt.setStyleSheet("QPushButton { background-color: #FFA500; color: white; padding: 10px 20px; border-radius: 5px; }")
        self.checkout_bt.setObjectName("checkout_bt")
        self.horizontalLayout_2.addWidget(self.checkout_bt)
        self.close_bt = QtWidgets.QPushButton(Information)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        self.close_bt.setFont(font)
        self.close_bt.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px 20px; border-radius: 5px; }")
        self.close_bt.setObjectName("close_bt")
        self.horizontalLayout_2.addWidget(self.close_bt)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(Information)
        QtCore.QMetaObject.connectSlotsByName(Information)

    def retranslateUi(self, Information):
        _translate = QtCore.QCoreApplication.translate
        Information.setWindowTitle(_translate("Information", "Notification"))
        self.nameLabel.setText(_translate("Information", "User Name"))
        self.timeInLabel.setText(_translate("Information", "Time In"))
        self.check_bt.setText(_translate("Information", "Check Attendance"))
        self.checkout_bt.setText(_translate("Information", "Check Out"))
        self.close_bt.setText(_translate("Information", "Close"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Information = QtWidgets.QWidget()
    ui = Ui_Information()
    ui.setupUi(Information)
    Information.show()
    sys.exit(app.exec_())

