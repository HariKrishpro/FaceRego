import os
import threading
import time

import cv2
import face_recognition
import numpy
import csv


def encoding_images(img):
    encoded_image = []
    for _ in img:
        _ = cv2.cvtColor(_, cv2.COLOR_BGR2RGB)
        _ = cv2.resize(_, (0, 0), None, 0.25, 0.25)
        encoded = face_recognition.face_encodings(_)[0]
        encoded_image.append(encoded)
    return encoded_image


def storing_to_csv(names):
    name = names

    with open('test_files/test.csv', 'w') as file:
        write = csv.writer(file)
        write.writerow(name)


def live_cam_to_check(cam):
    s = set()
    out = time.time() + 10
    print(out)
    while True:
        success, img_org = cam.read()
        img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)

        location_cam = face_recognition.face_locations(img)
        encode_cam = face_recognition.face_encodings(img, location_cam)

        for i, j in zip(encode_cam, location_cam):
            matches_location = face_recognition.face_distance(encoded_images, i)
            matches_encode = face_recognition.compare_faces(encoded_images, i)
            print(matches_location)
            match = numpy.argmin(matches_location)
            if matches_location[match] <= 0.6:
                s.add(names[match])
        cv2.imshow('live', img_org)
        cv2.waitKey(1)
        if time.time() > out:
            break

    return s


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

s = live_cam_to_check(cv2.VideoCapture(0))
storing_to_csv(list(s))
