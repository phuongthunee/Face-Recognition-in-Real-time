import sys
import cv2
import os
import tensorflow as tf
import numpy as np
import re
import csv
import sklearn
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer, QDate, QTime, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from GUI.main import Ui_FaceRecognition
from GUI.information import Ui_Information
from GUI.viewattendance import Ui_Attendance

from utils import face_preprocess
from load_models import load_mtcnn, load_mobilefacenet
import connect_DB

sys.path.append('C:/Users/lephu/OneDrive/Desktop/Face-Recognition-in-Real-time/')

sys.path.append('C:/Users/lephu/OneDrive/Desktop/Face-Recognition-in-Real-time/GUI/')

import configparser

conf = configparser.ConfigParser()
conf.read('C:/Users/lephu/OneDrive/Desktop/Face-Recognition-in-Real-time/config/main.cfg')

MODEL_PATH = conf.get("MOBILEFACENET", "MODEL_PATH")
VERIFICATION_THRESHOLD = float(conf.get("MOBILEFACENET", "VERIFICATION_THRESHOLD"))
FACE_DB_PATH = conf.get("MOBILEFACENET", "FACE_DB_PATH")

def draw_rect(faces, landmarks, names, sims, image):
    parent_data_names = []  # Store parent_data_name for each face
    for i, face in enumerate(faces):
        if names[i] == "unknown":
            parent_data_name = names[i]  # Assuming names[i] contains the parent_data_name
        else:
            employee_info = connect_DB.getEmployee(names[i])
            if employee_info is not None:
                parent_data_name = employee_info["name"]
            else:
                parent_data_name = "Unknown"

        parent_data_names.append(parent_data_name)  # Store the parent_data_name for each face

        size = cv2.getTextSize(parent_data_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x, y = int(face[0]), int(face[1])

        # Draw rectangle around the face
        x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, parent_data_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return parent_data_names

def feature_compare(feature1, feature2, threshold):
    dist = np.sum(np.square(feature1 - feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim

def load_data_from_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Skip header row
        for row in csvreader:
            data.append(row)
    return data

with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:
                
        class FaceRecognition(QWidget, Ui_FaceRecognition):
            #accept = pyqtSignal()
            def __init__(self):
                super(FaceRecognition, self).__init__()
                self.ui = Ui_FaceRecognition()
                self.ui.setupUi(self)
                current = QDate.currentDate()

                # Load the MTCNN detector and MobileFaceNet model
                self.mtcnn_detector = load_mtcnn(conf)  # Assuming you have the 'conf' object loaded
                self.sess, self.inputs_placeholder, self.embeddings = load_mobilefacenet(MODEL_PATH)
                self.faces_db = []
                print('Load models and sess DONE')
                
                #start video capture
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.video)
                self.timer.start(30) #fps = 30
                
                self.clock_timer = QTimer(self)
                self.clock_timer.timeout.connect(self.update_time)
                self.clock_timer.start(1000) 
                
                size = QApplication.desktop().screenGeometry()
                self.h = size.height()
                self.w = size.width()
                
                self.attend_arr = []
                self.id_set = set()
                self.frame_detect = 0
                currentDate = current.toString('dd/MM/yyyy')
                currentTime = datetime.now().strftime('%H:%M:%S')
                
                # Connect button click to a function
                self.ui.dateLabel.setText(currentDate)
                self.ui.timeLabel.setText(currentTime)
                
                self.start_webcam()
                self.recognition_thread = None
                
                self.face_detected = False #stop the infinite loop  
                
                self.ui.view.clicked.connect(self.show_attendance_history)
                self.check_in_times = {}

                            
            def load_faces(self, face_db_path):
                subset_embeddings = {}  # Dictionary to store embeddings by subset
                subdatasets = os.listdir(face_db_path)
                for subdataset in subdatasets:
                    subdataset_path = os.path.join(face_db_path, subdataset)
                    if os.path.isdir(subdataset_path):
                        image_files = os.listdir(subdataset_path)
                        for file in image_files:
                            image_path = os.path.join(subdataset_path, file)
                            if os.path.isfile(image_path):
                                raw_image = cv2.imread(image_path)
                                faces, landmarks = self.mtcnn_detector.detect(raw_image)
                                if faces is not None and len(faces) > 0:
                                    # Assuming only one face per image in the database
                                    bbox = faces[0, 0:4]
                                    points = landmarks[0, :].reshape((5, 2))
                                    face_image = face_preprocess.preprocess(raw_image, bbox, points, image_size='112,112')
                                    face_image = face_image - 127.5
                                    face_image = face_image * 0.0078125
                                    embedding = self.sess.run(self.embeddings, feed_dict={self.inputs_placeholder: np.expand_dims(face_image, axis=0)})
                                    embedding = sklearn.preprocessing.normalize(embedding).flatten()

                                    # Get the name from the file name (remove the extension and parent data name)
                                    name = os.path.splitext(file)[0]
                                    parent_data_name = os.path.basename(os.path.dirname(image_path))

                                    # Store the embedding in the corresponding subset in the dictionary
                                    if parent_data_name not in subset_embeddings:
                                        subset_embeddings[parent_data_name] = []
                                    subset_embeddings[parent_data_name].append({"name": name, "feature": embedding})

                # Store the subset embeddings in the main faces_db list
                for subset_name, embeddings_list in subset_embeddings.items():
                    representative_embedding = np.mean([entry["feature"] for entry in embeddings_list], axis=0)
                    representative_name = subset_name  # Use the subset name as the representative's name
                    self.faces_db.append({"name": representative_name, "feature": representative_embedding})

            @pyqtSlot()
            def on_update_db(self):
                print("Updating the database...")
            
            def on_logout(self):
                pass
            
            def on_login(self):
                pass
            
            def on_capture(self):
                pass
            
            def get_parent_data_name(self, name):
                employee_info = connect_DB.getEmployee(name)
                if employee_info is not None and "parent_data_name" in employee_info:
                    return employee_info["parent_data_name"]
                else:
                    return None
            
            def start_webcam(self):
                self.load_faces(FACE_DB_PATH)
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    print("Cannot open camera")
                    sys.exit()
                self.timer.start(100)  # Start the timer with a 100 ms interval (10 fps)
                self.timer.timeout.connect(self.video)

            @pyqtSlot()
            def video(self):
                ret, frame = self.camera.read()

                # Check if the frame is empty
                if frame is None:
                    print("Error: Could not read frame from camera")
                    return

                h, w, ch = frame.shape
                # Detect known faces in the frame
                image, faces, landmarks, face_names, sims = self.detect_known_faces(frame)
                ids = face_names  # Store the recognized face names in the 'ids' list

                if faces is None or face_names is None or landmarks is None or sims is None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytesPerLine = ch * w
                    qImg = QImage(rgb_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    self.ui.image_label.setPixmap(pixmap)
                    return

                # Draw boxes around detected faces and write their names
                for face_loc, name, sim in zip(faces, face_names, sims):
                    if sim > VERIFICATION_THRESHOLD:
                        # Face is known
                        label = "{} (Similarity: {:.2f})".format(name, sim)
                    else:
                        # Face is unknown
                        label = "Unknown"

                    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    x, y = int(face_loc[0]), int(face_loc[1])  # Corrected variable name
                    cv2.rectangle(frame, (x, y - size[1]), (x + size[0], y), (255, 167, 182), cv2.FILLED)  # Draw the rectangle on the original frame
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Write the label on the original frame

                    # Show information window for known faces with high similarity
                    if sim > VERIFICATION_THRESHOLD:
                        check = {'name': name, 'acc': sim, 'time': datetime.now()}
                        image_path = f"images/{check['name']}.jpg"  # Assuming the image file path is in the 'images' folder
                        qImg = self.displayImage(frame, h, w)  # Assign qImg here
                        self.confW = Information(check['name'], image_path, check['time'].strftime('%H:%M:%S'), parent=self)
                        self.timer.stop()
                        self.confW.show()
                        self.confW.setNoti(check, qImg)
                        self.confW.autoStart.connect(self.timer.start)
                        #self.attendance(name)

                # Convert frame to pixmap and set as label image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytesPerLine = ch * w
                qImg = QImage(rgb_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.ui.image_label.setPixmap(pixmap)

                # Display the resulting frame
                key = cv2.waitKey(1)

                # Exit on ESC key press
                if key == 27:
                    self.camera.release()
                    cv2.destroyAllWindows()
                    sys.exit()
                
            def detect_known_faces(self, image):
                faces, landmarks = self.mtcnn_detector.detect(image)
                ids = []
                sims = []
                emb_arrays = []

                if faces is None or len(faces) == 0:
                    # Return empty lists when no faces are detected
                    return image, None, None, [], []

                input_images = np.zeros((len(faces), 112, 112, 3))
                for i, face in enumerate(faces):
                    bbox = face[0:4]
                    points = landmarks[i, :].reshape((5, 2))
                    nimg = face_preprocess.preprocess(image, bbox, points, image_size='112,112')

                    nimg = nimg - 127.5
                    nimg = nimg * 0.0078125
                    input_images[i, :] = nimg

                with self.sess.as_default():
                    feed_dict = {self.inputs_placeholder: input_images}
                    emb_arrays = self.sess.run(self.embeddings, feed_dict=feed_dict)
                    emb_arrays = sklearn.preprocessing.normalize(emb_arrays)

                    for i, embedding in enumerate(emb_arrays):
                        embedding = embedding.flatten()
                        temp_dict = {}
                        for com_face in self.faces_db:
                            ret, sim = feature_compare(embedding, com_face["feature"], VERIFICATION_THRESHOLD)
                            temp_dict[com_face["name"]] = sim
                        dict = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
                        print(dict)
                        if dict == []:
                            id = "unknown"
                            sim = 0
                        else:
                            if dict[0][1] > VERIFICATION_THRESHOLD:
                                id = dict[0][0]
                                sim = dict[0][1]
                            else:
                                id = "unknown"
                                sim = 0

                        # Append the parent_data_name to the name if it's not "unknown"
                        if id != "unknown":
                            id = "{}".format(id, self.get_parent_data_name(id))

                        ids.append(id)
                        sims.append(sim)

                # Draw boxes around detected faces and write their names
                image = draw_rect(faces, landmarks, ids, sims, image)  # Pass 'landmarks' to draw_rect function

                return image, faces, landmarks, ids, sims

            #def displayImage(self, img, h, w):
            #    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #    qImg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
            #    return qImg
            def displayImage(self, img, h, w):
                height = int(self.h / 2)
                width =int(self.w / 2)
                # print(height,width)
                height, width, channel = height,width,3
                # height, width, channel = frame.shape
                # cv2.putText(image,'check',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
                step = channel * width
                # create QImage from image
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                qImg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                # 
                # show image in img_label
                self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
                # self.ui.image_label.setScaledContents(1)
                return qImg
            
            #@pyqtSlot(str)
            #def attendance(self, name):
            #    with open('attendance.csv', 'r+') as f:
            #        data = f.readlines()
            #        nameList = []
            #        for line in data:
            #            entry = line.split('-')
            #            nameList.append(entry[0])
            #        if name not in nameList:
            #            now = datetime.now()
            #            dtString = now.strftime('%H:%M:%S')
            #            f.writelines(f'\n{name}, {dtString}')

            @pyqtSlot() 
            def update_time(self):
                current_time = QTime.currentTime()
                current_date = QDate.currentDate()
                time_str = current_time.toString('hh:mm:ss')
                date_str = current_date.toString('dd/MM/yyyy')
                self.ui.timeLabel.setText(time_str)
                self.ui.dateLabel.setText(date_str) 
            
            @pyqtSlot()
            def return_face_detected(self):
                self.face_detected = False

            def show_information_window(self, name, image, timeIn):
                self.info_window = Information(name, image, timeIn)
                self.info_window.setAttribute(Qt.WA_DeleteOnClose)
                self.info_window.show()
                self.info_window.closeEvent = lambda event: self.return_face_detected()
                self.face_detected = True   
                self.info_window.returnToIdentification.connect(self.handle_return_to_identification)

            def handle_return_to_identification(self):
                self.info_window.close()  # Close the Information window
                self.timer.start()
                    
            def show_attendance_history(self):
                self.attendance_history_window = AttendanceHistory()
                self.attendance_history_window.show()
        
        class Information(QWidget, Ui_Information):
            autoRecord = pyqtSignal()
            autoStart = pyqtSignal()
            returnToIdentification = pyqtSignal()

            def __init__(self, name, image_path, timeIn, timeout=5, parent=None):
                super(Information, self).__init__()
                self.setupUi(self)
                self.nameLabel.setText(str(name))

                self.close_bt.clicked.connect(self.acceptContent)
                self.time_to_wait = timeout
                self.timer = QtCore.QTimer(self)
                self.timer.setInterval(1000)
                self.timer.timeout.connect(self.changeContent)
                self.timer.start()
                
                self.check_bt.clicked.connect(self.save_attendance)
                self.checkout_bt.clicked.connect(self.checkout)
                self.parent = parent
                if name in self.parent.check_in_times:
                    self.checkout_bt.setEnabled(True)
                else:
                    self.checkout_bt.setEnabled(False)

                self.timeInLabel.setText(timeIn)
                
            def save_attendance(self):
                name = self.nameLabel.text()
                timeIn = self.timeInLabel.text()
                filename = 'attendance.csv'
                headers = ["Name", "Time", "Date", "Action", "Duration"]

                # Check if file exists and has content
                file_exists = os.path.isfile(filename)
                if not file_exists or os.stat(filename).st_size == 0:
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)

                # If the person has not checked in yet, check them in
                if self.nameLabel.text() not in self.parent.check_in_times:
                    self.parent.check_in_times[self.nameLabel.text()] = datetime.now()
                    action = "Check In"
                    duration = ""

                    with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, timeIn, datetime.now().date(), action, duration])

                    # Display the notification
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Check In - DONE!")
                    msg.setWindowTitle("Notification")
                    msg.exec_()

                    # Close the information window
                    self.close()
                else:
                    # Person is already checked in; display a message box
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("You are already checked in. Please check out first!")
                    msg.setWindowTitle("Warning")
                    msg.exec_()
    
            def checkout(self):
                if self.nameLabel.text() in self.parent.check_in_times:
                    check_in_time = self.parent.check_in_times.pop(self.nameLabel.text())
                    duration = datetime.now() - check_in_time
                    # Display the duration on the screen
                    self.timeInLabel.setText(f"Duration: {duration}")

                    # Save the "Check Out" action and the duration to the CSV file
                    name = self.nameLabel.text()
                    timeOut = datetime.now().strftime('%H:%M:%S')
                    filename = 'attendance.csv'
                    action = "Check Out"
                    duration_str = str(duration)

                    with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, timeOut, datetime.now().date(), action, duration_str])
                        
                    # Display the notification
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Check Out - DONE!")
                    msg.setWindowTitle("Notification")
                    msg.exec_()

                    # Close the information window
                    self.close()
                else:
                    # Person has not checked in yet; display a message box
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Please check in first before checking out!")
                    msg.setWindowTitle("Warning")
                    msg.exec_()

            def setNoti(self, check, qImg):
                self.imageLabel.setPixmap(QPixmap.fromImage(qImg))

            def changeContent(self):
                self.close_bt.setText("Close ({0})".format(self.time_to_wait))
                if self.time_to_wait == 0:
                    self.close()
                    self.autoRecord.emit()
                    self.autoStart.emit()
                    self.timer.stop()
                    self.returnToIdentification.emit()  # Emit the signal when the timeout occurs
                self.time_to_wait -= 1
                print(self.time_to_wait)

            def acceptContent(self):
                self.close()
                self.autoRecord.emit()
                self.autoStart.emit()
                self.timer.stop()
                self.close()
            
            def declineContent(self):
                self.close()
                self.autoStart.emit()
                self.timer.stop()

                text = 'Full name: ' + self.nameLabel.text()
                self.nameLabel.setText(text)
                self.nameLabel.setWordWrap(True)
                self.nameLabel.setText('\u200b'.join(text))

                text = 'Time Check-in: ' + self.timeInLabel.text()
                self.timeInLabel.setText(text)
                self.timeInLabel.setWordWrap(True)
                self.timeInLabel.setText('\u200b'.join(text))
                
        class AttendanceHistory(QtWidgets.QWidget, Ui_Attendance):
            def __init__(self):
                super(AttendanceHistory, self).__init__()
                self.setupUi(self)
                self.load_attendance()

            def load_attendance(self):
                data = load_data_from_csv('attendance.csv')

                self.tableWidget.setRowCount(len(data))
                self.tableWidget.setColumnCount(5)
                self.tableWidget.setHorizontalHeaderLabels(["Name", "Time", "Date", "Action", "Duration"])

                for row, record in enumerate(data):
                    for col, item in enumerate(record):
                        cell = QtWidgets.QTableWidgetItem(str(item))
                        self.tableWidget.setItem(row, col, cell)
                self.tableWidget.resizeColumnsToContents()

                        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = FaceRecognition()
    mainWindow.show()
    sys.exit(app.exec_())