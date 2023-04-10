import cv2
import face_recognition 

img = cv2.imread("phuongthunee.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encodings = face_recognition.face_encodings(rgb_img)
if len(img_encodings) > 0:
    img_encoding = img_encodings[0]
else:
    print("No face detected in the image.")


img2 = cv2.imread("images/phuongthunee.jpg")
rgb_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encodings2 = face_recognition.face_encodings(rgb_img)
if len(img_encodings2) > 0:
    img_encodings2 = img_encodings2[0]
else:
    print("No face detected in the image.")

result = face_recognition.compare_faces([img_encoding], img_encodings2)
print("Result: ", result)

cv2.imshow("phuongthunee", img)
cv2.imshow("phuongthunee", img2)
cv2.waitKey(0)
