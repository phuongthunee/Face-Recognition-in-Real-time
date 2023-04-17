import sys
import cv2
from datetime import datetime
from simple_facerec import SimpleFacerec
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, QDate, QTime
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, QMessageBox, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from GUI.main import Ui_FaceRecognition

class FaceRecognition(QMainWindow, Ui_FaceRecognition):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        self.setupUi(self)
        current = QDate.currentDate()
        
        currentDate = current.toString('dd/MM/yyyy')
        currentTime = datetime.now().strftime('%H:%M:%S')
        #currentName = 
        
        self.dateLabel.setText(currentDate)
        self.timeLabel.setText(currentTime)
        #self.nameLabel.setText(currentName)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000) #the timer is started with an interval of 1000 ms (1 second) to update the time label every second

        #encode faces from a folder
        self.image = None
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images('images/')
        
        #webcam
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Cannot open camera")
            sys.exit()
        
        #start video capture
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.video)
        self.timer.start(30) #fps = 30
        
    @pyqtSlot() 
    def video(self):
        ret, frame = self.camera.read()

        #check if the frame is empty
        if not ret:
            print("Error: Could not read frame from camera")
            return

        #resize
        #resized_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
        
        #detect Faces
        face_locations, face_names = self.sfr.detect_known_faces(frame)

        #draw boxes around detected faces and write their names
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            self.attendance(name)
        
        #convert frame to pixmap and set as label image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytesPerLine = ch * w
        qImg = QImage(rgb_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.videoLabel.setPixmap(pixmap)
        
        #display the resulting frame
        key = cv2.waitKey(1)
        #exit on ESC key press
        if key == 27:
            self.camera.release()
            cv2.destroyAllWindows()
            sys.exit()

    @pyqtSlot(str)
    def attendance(self, name):
        with open('attendance.csv', 'r+') as f:
            data = f.readlines()
            nameList = []
            for line in data:
                entry = line.split('-')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name}, {dtString}')
                QMessageBox.information(self, 'Attendance', f'{name} has been marked as present!')

    @pyqtSlot() 
    def update_time(self):
        current_time = QTime.currentTime()
        current_date = QDate.currentDate()
        time_str = current_time.toString('hh:mm:ss')
        date_str = current_date.toString('dd/MM/yyyy')
        self.timeLabel.setText(time_str)
        self.dateLabel.setText(date_str) 
         
app = QApplication(sys.argv)
mainWindow = FaceRecognition()

#start video capture
timer = QTimer(mainWindow)
timer.timeout.connect(mainWindow.video)
timer.start(30)

mainWindow.show()
sys.exit(app.exec_())