import win32gui
import win32ui
import win32con
import win32api
from time import time

import cv2
import numpy

class ScreenCapture:
    'help to capture screen'

    __screenSize = (0, 0)
    __hWndDesktop = 0
    __wDC = 0
    __dcObj = None
    __cDC = None
    __buffer = None
    

    def __init__(self):
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        self.__screenSize = (width, height)

        
        self.__wDC = win32gui.GetWindowDC(self.__hWndDesktop)
        self.__dcObj = win32ui.CreateDCFromHandle(self.__wDC)
        self.__cDC = self.__dcObj.CreateCompatibleDC()

        self.__buffer = win32ui.CreateBitmap()
        self.__buffer.CreateCompatibleBitmap(self.__dcObj, width, height)
        self.__cDC.SelectObject(self.__buffer)
    
    def __del__(self):
        self.__dcObj.DeleteDC()
        self.__cDC.DeleteDC()
        win32gui.ReleaseDC(self.__hWndDesktop, self.__wDC)
        win32gui.DeleteObject(self.__buffer.GetHandle())

    @property
    def screenSize(self):
        return self.__screenSize
    
    def capture(self):
        self.__cDC.BitBlt((0, 0), self.__screenSize, self.__dcObj, (0, 0), win32con.SRCCOPY)
        signedIntsArray = self.__buffer.GetBitmapBits(True)
        im_opencv = numpy.frombuffer(signedIntsArray, dtype = 'uint8')
        w, h = self.__screenSize
        im_opencv.shape = (h, w, 4)
        cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)

        return im_opencv


if __name__=='__main__':
    capture = ScreenCapture()
    print(capture.screenSize)
    beginTime = time()
    img = capture.capture()
    print('take screen use {:.2f}ms'.format((time() - beginTime) * 1000))

    cv2.imshow('preview', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
