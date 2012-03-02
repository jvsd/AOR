from numpy import *
from matplotlib.pyplot import *
import time
from sklearn import decomposition, svm
import cv
import Image
#Used with human input to give selected areas a rating of true or false to be validated later

def GetData(src):
    xt = 200
    yt = 200
    xo = 0
    yo = 0
    targets = zeros(640)
    cropped = [None]*640
    for i in range(0,640):
        cropped[i] = cv.CreateImage((400,400) , cv.IPL_DEPTH_8U, 3)
        cv.SetImageROI(src,(xo,yo,400,400))
        cv.Copy(src,cropped[i], None)
        cv.ResetImageROI(src)
        cv.ShowImage("Test",cropped[i])
        key = cv.WaitKey(0)
        #var = input("True or false?")
        if (ord('0') == key):
            print 'Mark True'
        else:
            print 'Mark False'
            targets[i] = 1
            
        if xo == 8000-400:
            xo = 0
            yo = yt+yo
        else:
            xo = xo+xt
            
    return cropped,targets
      
source = 'Data/MLtest1.jpg'
src = cv.LoadImage(source, cv.CV_LOAD_IMAGE_UNCHANGED)


            

(cropped,targets) = GetData(src)
print targets

#cv.ShowImage("test",cropped)
#cv.WaitKey()