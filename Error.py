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
            cv.ShowImage("test",testset[b+temp])
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

def shuffle_unison(a, b):
    rng_state = random.get_state()
    random.shuffle(a)
    random.set_state(rng_state)
    random.shuffle(b)
    
data = load('Data/datatest.npy')
target = load('Data/targettest.npy')

testdata = load('Data/datavalid.npy')
testtarget = load('Data/targetvalid.npy')


testsource = 'Data/test.png'
test = cv.LoadImage(testsource, cv.CV_LOAD_IMAGE_UNCHANGED)

(test1,tempt) = getObjectFeatures(test,0)

#Run PCA on Training Data
n_components = 100;
pca = PCA(n_components)
pca.fit(data)

#Get total variance explained
components = pca.explained_variance_ratio_
total = components.sum()
print 'done PCA'
print total

#Transform Training and Test Data
dataPCA = pca.transform(data)
testDataPCA = pca.transform(testdata)

ctemp = linspace(.000000001,1,num = 100)
TrainingScore = [None]*100
ValidationScore = [None]*100
Error = [None]*100
Errortest = [None]*100
for i in range(0,100):            
#Create and Train SVM
    print ctemp[i]
    clf = svm.SVC(C = 200, kernel = 'rbf', degree=3,gamma=0.0, coef0=0.0, shrinking=True,probability=False,tol= ctemp[i])
    print target.shape
    print 'shape'
    clf.fit(dataPCA, target)

#Optional Shuffle of Data
#shuffle_unison(dataPCA,target)

#Training Score
    TrainingScore[i] = clf.score(dataPCA,target)
#Validation Score
    ValidationScore[i] = clf.score(testDataPCA,testtarget)
    
    results = clf.predict(testDataPCA)
    Diff = (results - testtarget)
    Error[i] = dot(Diff,Diff)/len(testtarget)
    #for x in range(0,len(testtarget)):
        #if(results[x] == 0):
            #print x
    resultstest = clf.predict(dataPCA)
    Difftest = (resultstest-target)
    Errortest[i] = dot(Difftest,Difftest)/len(target)

fig = figure()
fig.hold(True)
plot(ctemp,Error)
plot(ctemp,Errortest,'r')
xlabel('Stopping Criterion')
ylabel('Mean Squared Error')
title('MSE vs Stopping Criterion')
savefig("Data/Error.jpg",format = "jpg")

print 'Mean Validation Squared Error'
print Error
print 'Mean Test Squared Error'
print Errortest
print 'Training Score'
print TrainingScore

print 'Validation Score'
print ValidationScore