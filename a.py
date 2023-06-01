import os
import cv2
import glob


import cv2

path = "IP-C980-50_20230524103512.avi"

output_folder = '1/'


vc = cv2.VideoCapture(path)
fps = vc.get(cv2.CAP_PROP_FPS)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
video = []

for idx in range(frame_count):
    vc.set(1, idx)
    ret, frame = vc.read()
    height, width, layers = frame.shape
    size = (width, height)

    if frame is not None:
        file_name = '{}{:08d}.jpg'.format(output_folder,idx)
        cv2.imwrite(file_name, frame)

    print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
vc.release()
