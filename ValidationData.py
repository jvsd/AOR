from numpy import *
from matplotlib.pyplot import *
import time
from sklearn import decomposition, svm
#from scikits.learn import datasets
#from scikits.learn import decomposition
#from scikits.learn.decomposition import pca
import cv
import Image

#Used to get validation data from an image


def GetData(src):
    xt = 200
    yt = 200
    xo = 400
    yo = 2700
    number = 30
    targets = zeros(number)
    cropped = [None]*number
    for i in range(0,number):
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

def shuffle_unison(a, b):
    rng_state = random.get_state()
    random.shuffle(a)
    random.set_state(rng_state)
    random.shuffle(b)
    
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
            cv.ShowImage("test",testset[b+temp])
            cv.WaitKey()
            (keypoints, descriptors) = cv.ExtractSURF(testset[b+temp],None,cv.CreateMemStorage(),(0,1,3,1))
            if (len(keypoints) < 10):
                keypoints = [None]*10
                for x in range(0,10):
                    keypoints[x] = zeros(69)
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
        
        
source = 'Data/MLtest3.jpg'
src = cv.LoadImage(source, cv.CV_LOAD_IMAGE_UNCHANGED)
        
#im1 = Image.open(source)
#im2 = Image.open('/Users/jamesd/Desktop/5_5.jpg')
#test = Image.open('/Users/jamesd/Desktop/6.jpg')
testsource = 'Data/test.png'
#test = cv.LoadImage(testsource, cv.CV_LOAD_IMAGE_UNCHANGED)




#data1 = getObjectFeatures(im1,5)
#data2 = getObjectFeatures(im2,5)
#test1 = getObjectFeatures(test,5)
#test1 = asarray(test1)

#target = [1]*25
#target = [0]*25 + target
#targettest=[0]*25
#targettest=asarray(targettest)

#print shape(data1) #
#data = concatenate((data1,data2))


#target = asarray(target)

(cropped,targets) = GetData(src)
data =[]
target = [None]

print len(cropped)
print targets.shape
print type(cropped)     
for i in range(0,len(cropped)):
    print i
    input = targets[i]
    if (i == 0):
        (data,target) = getObjectFeatures(cropped[i],input)
    else:
        (tempdata,temptargets) = getObjectFeatures(cropped[i],input)
        data = concatenate((data,tempdata))
        target = concatenate((target,temptargets))
    
#shuffle_unison(data,target)
#print target

save('Data/dataTemp.npy',data)
save('Data/targetTemp.npy',target)

#datatemp = asarray(datatemp)
#datatemp = ([datatemp,datatemp[0:]])
#print data.shape
#tic = time.clock()
#clf = svm.SVC()
#clf.fit(data, target)
#toc = time.clock()  
#print "Time to Train..."
#print toc-tic

#print clf.score(test1,targettest)
#for i in range(0,25):
 #   print clf.predict(test1[i])


#for i in range(0,len(keypoints)-1):
#    point = keypoints[i][0]
    #print keypoints[i]
#    cv.Circle(im2,point,35,(0,0,255,0),1,8,0)
    
#cv.SaveImage('/Users/jamesd/Desktop/testout.jpg',im2)

#print descriptors[3]
  
#print toc-tic
