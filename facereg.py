import sys
import cv2
import os
from datetime import datetime
from detect_faces import detectFaces
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer, QDate, QTime, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QImage
from GUI.main import Ui_FaceRecognition
from GUI.information import Ui_Information

class FaceRecognition(QMainWindow, Ui_FaceRecognition):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        self.setupUi(self)
        current = QDate.currentDate()
        
        currentDate = current.toString('dd/MM/yyyy')
        currentTime = datetime.now().strftime('%H:%M:%S')
        
        self.dateLabel.setText(currentDate)
        self.timeLabel.setText(currentTime)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000) #the timer is started with an interval of 1000 ms (1 second) to update the time label every second

        self.face_detected = False #stop the infinite loop
        
        #encode faces from a folder
        self.image = None
        self.sfr = detectFaces()
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
        
        #split the frame into individual color channels
        b, g, r = cv2.split(frame)
        #merge the channels back together
        frame = cv2.merge((b, g, r))
        #check if the frame is empty
        if not ret:
            print("Error: Could not read frame from camera")
            return
        
        if self.face_detected:
            return
        
        #detect faces
        face_locations, face_names = self.sfr.detect_known_faces(frame)

        #draw boxes around detected faces and write their names
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 206, 49), 4)
            self.attendance(name)
            now = datetime.now().strftime('%H:%M:%S')
            self.show_information_window(name, frame[y1:y2, x1:x2], now)
            self.face_detected = True
            break
        
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

    @pyqtSlot() 
    def update_time(self):
        current_time = QTime.currentTime()
        current_date = QDate.currentDate()
        time_str = current_time.toString('hh:mm:ss')
        date_str = current_date.toString('dd/MM/yyyy')
        self.timeLabel.setText(time_str)
        self.dateLabel.setText(date_str) 
    
    @pyqtSlot()
    def return_face_detected(self):
        self.face_detected = False

    def show_information_window(self, name, image, timeIn):         
        self.info_window = Information(name, image, timeIn)
        self.info_window.setAttribute(Qt.WA_DeleteOnClose)
        self.info_window.show()
        self.info_window.closeEvent = lambda event: self.return_face_detected()
                        
class Information(QMainWindow, Ui_Information):
    def __init__(self, name, image, timeIn):
        super(Information, self).__init__()
        self.setupUi(self)
        self.nameLabel.setText(name)
        
        if os.path.exists(image):
            raw_image = cv2.imread(image)
            h, w, ch = raw_image.shape
            qImg = QImage(raw_image.data, w, h, ch * w, QImage.Format_BGR888)
            image = QPixmap.fromImage(qImg)
        else:
            #fall back to the thumbnail image
            image = os.path.join('images/', f'{name}.jpg')
            raw_image = cv2.imread(image)
            h, w, ch = raw_image.shape
            bytesPerLine = ch * w
            qImg = QImage(raw_image.data, w, h, bytesPerLine, QImage.Format_BGR888)
            image = QPixmap.fromImage(qImg)
        self.imageLabel.setPixmap(image.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        self.timeInLabel.setText(timeIn)                              
                      
app = QApplication(sys.argv)
mainWindow = FaceRecognition()

#start video capture
timer = QTimer(mainWindow)
timer.timeout.connect(mainWindow.video)
timer.start(30)

mainWindow.show()
sys.exit(app.exec_())