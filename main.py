from time import time
from capture import ScreenCapture
import detect
import os
import cv2

snap = ScreenCapture()

while 1:
    beginTime = time()
    
    img = snap.capture()
    r = detect.detectImage(img)
    print(r)

    elapsed = (time() - beginTime) * 1000
    print('take screen use {:.2f}ms'.format(elapsed))

    os.system('title {} / {:.2f} / {:.2f}'.format(r, elapsed, 1000 / elapsed))

