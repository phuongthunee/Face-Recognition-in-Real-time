import cv2
from simple_facerec import SimpleFacerec 

#encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = camera.read()

    #check if the frame is empty
    if not ret:
        print("Error: Could not read frame from camera")
        break

    #detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    #draw boxes around detected faces and write their names
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    #print frame size
    print(frame.shape)
    
    #display the resulting frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    #exit on ESC key press
    if key == 27:
        break

    
camera.release()
cv2.destroyAllWindows()
    