import os
import threading

import cv2
import face_recognition
import numpy


def encoding_images(images):
    encoded_image = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        encoded = face_recognition.face_encodings(img)[0]
        encoded_image.append(encoded)
    return encoded_image


def showing_cam_live():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()

        cv2.imshow('vid',frame)
        cv2.waitKey(1)


# storing for read image
images = []

# storing for image names
names = []

# storing for image path
images_path = os.listdir('test_photos')

for img in images_path:
    current_img = cv2.imread(f'test_photos/{img}')
    images.append(current_img)
    names.append(img.split('.')[0])

print('Encoding Starts....')
encoded_images = encoding_images(images)
print('Encoding Finished....')

cam = cv2.VideoCapture(0)
temp = 0

flag = True

while True:
    success, img_org = cam.read()
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0,0),None,0.25,0.25)

    location_cam = face_recognition.face_locations(img)
    encode_cam = face_recognition.face_encodings(img,location_cam)


    for i,j in zip(encode_cam,location_cam):
        matches_location = face_recognition.face_distance(encoded_images, i)
        matches_encode = face_recognition.compare_faces(encoded_images, i)
        print(matches_location)
        match = numpy.argmin(matches_location)
        if matches_location[match]<=0.6:
            print(names[match])

    cv2.imshow('livecam',img_org)
    cv2.waitKey(1)