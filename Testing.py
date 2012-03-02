from numpy import *
from matplotlib.pyplot import *
import time
from sklearn import decomposition, svm
from sklearn.decomposition import PCA
import cv
import Image
#simple file that reads an image and draws circles around extracted SURF objects

fullimageinput = 'Data/MLtest1.jpg'
fullimage = cv.LoadImage(fullimageinput, cv.CV_LOAD_IMAGE_GRAYSCALE)
fullimagec = cv.LoadImage(fullimageinput, cv.CV_LOAD_IMAGE_UNCHANGED)

(keypoints, descriptors) = cv.ExtractSURF(fullimage,None,cv.CreateMemStorage(),(0,1500,3,1))

point = [None]*len(keypoints)
for i in range(0,len(keypoints)-1):
    point[i] = keypoints[i][0]
    cv.Circle(fullimagec,point,35,(0,0,255,0),1,8,0)
    point[i][0] = point[i][0] - 200
    point[i][1] = point[i][1] + 200
    
    
cv.SaveImage('/Users/jamesd/Desktop/TESTSURF.jpg',fullimagec)

