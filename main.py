import cv2
import face_recognition
import numpy as np

img = face_recognition.load_image_file("test_photos/depp-square.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (960, 960))

img_test = face_recognition.load_image_file("test_photos/depp-square-test.jpg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
img_test = cv2.resize(img_test, (960, 960))

face_location = face_recognition.face_locations(img)[0]
face_encode = face_recognition.face_encodings(img)[0]


cv2.rectangle(img=img,
              pt1=(face_location[0], face_location[3]),
              pt2=(face_location[1], face_location[2]),
              color=(255, 0, 255),
              thickness=5)


face_location_test = face_recognition.face_locations(img_test)[0]
face_encode_test = face_recognition.face_encodings(img_test)[0]

cv2.rectangle(img=img_test,
              pt1=(face_location_test[0], face_location_test[3]),
              pt2=(face_location_test[1], face_location_test[2]),
              color=(255, 0, 255),
              thickness=5)

result = face_recognition.compare_faces([face_encode], face_encode_test)
print(result)

cv2.imshow('depp-square-test', img_test)
cv2.waitKey(0)
