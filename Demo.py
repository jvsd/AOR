from numpy import *
from matplotlib.pyplot import *
import time
from sklearn import decomposition, svm
from sklearn.decomposition import PCA
import cv
import Image

def getObjectFeatures(image,targetsin):


    size = (image.width, image.height)
    data = image.tostring()
    im1 = Image.fromstring("RGB",size,data)
     
#Rotating an image using PIL
    rim1 = im1.rotate(72)
    rim2 = im1.rotate(72*2)
    rim3 = im1.rotate(72*3)
    rim4 = im1.rotate(72*4)
    rim5 = im1
    
    cvrim=[None]*5
    
    cvrim[0] = cv.CreateImageHeader(rim1.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cvrim[0], rim1.tostring())
        
    cvrim[1] = cv.CreateImageHeader(rim2.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cvrim[1], rim2.tostring())
        
    cvrim[2] = cv.CreateImageHeader(rim3.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cvrim[2], rim3.tostring())
        
    cvrim[3] = cv.CreateImageHeader(rim4.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cvrim[3], rim4.tostring())
        
    cvrim[4] = cv.CreateImageHeader(rim5.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cvrim[4], rim5.tostring())
    
    #rim1.show()
    #cv.ShowImage("test",cvrim[0])
    #cv.WaitKey()
    
    #im = cv.LoadImageM('/Users/jamesd/Desktop/test.jpg',cv.CV_LOAD_IMAGE_GRAYSCALE)
    #im2 = cv.LoadImage('/Users/jamesd/Desktop/test.jpg')
    bwcvrim = [None]*5
    testset = [None]*25
    data =[None]*25
    
    for i in range(0,5):
    #tic = time.clock()
        bwcvrim[i] = cv.CreateImage((cvrim[i].width,cvrim[i].height) , cv.IPL_DEPTH_8U, 1)
        #cv.CvtColor(cvrim[i],bwcvrim[i], cv.CV_RGB2GRAY)
        cv.Split(cvrim[i],None,None,bwcvrim[i],None)
        cv.Threshold(bwcvrim[i],bwcvrim[i],60,255,cv.CV_THRESH_BINARY_INV)
        for b in range(0,5):
            temp = i*5
            testset[b+temp] = cv.CreateImage((bwcvrim[i].width,bwcvrim[i].height) , cv.IPL_DEPTH_8U, 1)
            adj = array([5,15,25,35,45])
            cv.Smooth(bwcvrim[i],testset[b+temp],cv.CV_GAUSSIAN,adj[b],5)
            #cv.ShowImage("test",testset[b+temp])
            #cv.WaitKey()
            (keypoints, descriptors) = cv.ExtractSURF(testset[b+temp],None,cv.CreateMemStorage(),(0,1,3,1))
            if (len(keypoints) < 10):
                keypoints = [None]*10
                for x in range(0,10):
                    keypoints[x] = zeros(69)
                    print "not enough keypoints"
            else:
                for x in range(0,len(keypoints)):
                    keypoints[x] = keypoints[x] + tuple(descriptors[x])
               
            data[b+temp] = keypoints
            
            keypoints = [None]
           
            
            
        #print descriptors[0]
        #data[b+temp] = 
        #keypoints[0] = keypoints[0] + descriptors[0]
        
        #print keypoints[0].append(descriptors[0])
    datapca = [None]*25
    dataout = [None]*25
    for i in range(0,25):
        sortby = "hessian"        
        #keypoints[i].sort(key=itemgetter(sortby))
        temp = data[i]
        #print temp
        data[i] = sorted(temp, key = lambda out: out[4])
        data[i].reverse()
        datapca[i] = data[i][0:10]
       
        #print datapca[0][9][4:]
        dataout[i] = datapca[i][0][5:] + datapca[i][1][5:] + datapca[i][2][5:] + datapca[i][3][5:] + datapca[i][4][5:] + datapca[i][5][5:] + datapca[i][6][5:] + datapca[i][7][5:] + datapca[i][8][5:] + datapca[i][9][5:]
        if (len(dataout[i]) < 640):
            dataout[i] = zeros(640)
         
    datax = asarray(dataout)
    #print 'length'
    #print len(datax[2])
    datatemp = zeros((25,640))
    for i in range(0,25):
        #print i
        datatemp[i] = datax[i][0:]
    temptarget = [targetsin]*25
    return datatemp,temptarget


data1 = load('Data/datatest.npy')
target1 = load('Data/targettest.npy')

n_components = 100;
pca = PCA(n_components)
pca.fit(data1)
dataPCA = pca.transform(data1)
clf = svm.SVC(C = 200, kernel = 'rbf', degree=3,gamma=0.0, coef0=0.0, shrinking=True,probability=False,tol= .2222)
clf.fit(dataPCA, target1)

fullimageinput = 'Data/MLtest1.jpg'
fullimage = cv.LoadImage(fullimageinput, cv.CV_LOAD_IMAGE_GRAYSCALE)
fullimagec = cv.LoadImage(fullimageinput, cv.CV_LOAD_IMAGE_UNCHANGED)

(keypoints, descriptors) = cv.ExtractSURF(fullimage,None,cv.CreateMemStorage(),(0,2500,3,1))

point = [None]*len(keypoints)
for i in range(0,len(keypoints)-1):
    point[i] = keypoints[i][0]
    cv.Circle(fullimagec,(point[i][0],point[i][1]),35,(0,0,255,0),1,8,0) 
    tempx = point[i][0]
    tempy = point[i][1]
    
    tempx = tempx - 200
    tempy = tempy - 200
    if (tempx < 0):
        tempx = 0
    
    if (tempy < 0):  
        tempy = 0
        
    if (tempx > 7600):
        tempx = 7600
        
    if (tempy > 3600):  
        tempy = 3200
        
    point[i] = (tempx,tempy)
xsize = len(point)-1
cropped = [None]*xsize
print 'here'
for i in range(0,len(point)-1):
        print point
        cropped[i] = cv.CreateImage((400,400) , cv.IPL_DEPTH_8U, 3)
        cv.SetImageROI(fullimagec,(point[i][0],point[i][1],400,400))
        cv.Copy(fullimagec,cropped[i], None)
        cv.ResetImageROI(fullimagec)
        cv.ShowImage("Test",cropped[i])
        key = cv.WaitKey(0)

data = [None]*len(cropped)
target = [None]*len(cropped)
for i in range(0,len(cropped)):
    print i
    input = 0
    if (i == 0):
        (data,target) = getObjectFeatures(cropped[i],input)
    else:
        (tempdata,temptargets) = getObjectFeatures(cropped[i],input)
        data = concatenate((data,tempdata))
        target = concatenate((target,temptargets))




inputPCA = pca.transform(data)

results = clf.predict(inputPCA)
ysize = len(results)/25
tempxx = -1
tempx = -2
for i in range(0,len(results)):
    if (results[i] == 0):
        tempx = int(round(i/25))
        if (tempx != tempxx):
            cv.ShowImage("Output",cropped[tempx])
            key = cv.WaitKey()
            tempxx = tempx


