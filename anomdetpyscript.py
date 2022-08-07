import numpy as np 
import pandas as pd
import os
import hdbscan
import matplotlib.pyplot as plt
import cv2
import sklearn
import xgboost as xgb
import pywt
import skimage.measure
import mahotas as mt
from skimage.filters import laplace, sobel, gabor_kernel, prewitt_h,prewitt_v
#from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from sklearn.decomposition import PCA
from skimage.transform import resize
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


class Anomdet:
    
    def __init__(self):
        print('training instance start')

                   
        
    def getdata(self,path,A):                                 # feature extraction. Array A acts like feature selector
        images = os.listdir(path)
        features=[]
        for img in images:
            feature = []
            a = plt.imread(path+img,0)
            img1 = cv2.resize(a, (224, 224))
#            img_n1 = np.array(img1).copy()

            b = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            if A[0] == 1:

                # Edge detection features using filters like laplace,sobel and prewitt 

                lap_feat = laplace(b)
                sob_feat = sobel(b)

                hpre_feat = prewitt_h(b)
                vpre_feat = prewitt_v(b)
                feature.extend([lap_feat.mean(),lap_feat.var(),np.amax(lap_feat),
                                sob_feat.mean(),sob_feat.var(),np.max(sob_feat),hpre_feat.mean(),hpre_feat.var(),np.max(hpre_feat),vpre_feat.mean(),vpre_feat.var(),np.max(vpre_feat)])



            if A[1] == 1:

                # Haralick features

                textures = mt.features.haralick(img1.astype(int))          
                for i in textures:
                    feature.extend(i) 



            if A[2] == 1:

                # Oriented FAST and rotated BRIEF(ORB) features which replace the SIFT and SURF in recent versions of python

                orb = cv2.ORB_create()
                # find the keypoints with ORB
                kp = orb.detect(img1,None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img1, kp)
        #         for d in des:
        #              feature.extend([d.mean(),d.var(),np.amax(d)])
                feature.extend([len(kp)])


            if A[3] == 1:

                # Histogram of Oriented Gradients(HOG)

                fd, hog_image = hog(b, orientations=4, pixels_per_cell=(14,14),cells_per_block=(1,1), visualize=True)
                c=0
                for i in fd:
                    b[c] = i/np.linalg.norm(fd)
                    c +=1
#                         feature.extend(i)
                feature.extend(c)    




            if A[4] == 1:

                # DWT of the image and using the energy of patches of the image

                coeffs2 = pywt.dwt2(b, 'db1')
                LL, (LH, HL, HH) = coeffs2
                for i in [LL,LH,HL,HH]:
                    op=skimage.measure.block_reduce(i, (14,14), np.linalg.norm)
                    op1 = i/np.linalg.norm(op)
                    feature.append(op1)


#                 if A[5] == 1:     # Removed from opensource in the recent versions of cv2

#                     sift = cv2.SIFT_create()
#                     keypoints, descriptors= sift.detectAndCompute(b, None)
#                     for k in descriptors:
#                         feature.extend([k.mean(),k.var(),np.amax(k)])    



#                 if A[6] == 1:       # Fourier data
#                     fft_image = np.fft.fft(image)
#                     fft_data.append(fft_image)

#                     freq  = np.fft.fftfreq(np.array(image).shape[-1], d=0.01)
#                     fft_freq.append(freq )

#                     fft_ps = np.abs(fft_image)**2
#                     power_spec.append(fft_ps)

#                     feature.extend([fft_data, fft_freq, power_spec])


            features.append(feature)
    
        return features



    # Training SVDD       
    def svdtrain(self,all_features):
        self.osvm_model = svm.OneClassSVM(nu=0.001,kernel='sigmoid')
        self.osvm_model.fit(all_features)
        
        
    # Training GLOSH    
    def glosh(self):
        self.clusters = hdbscan.HDBSCAN(min_cluster_size=100)
        self.clusters.fit(self.all_features)
        
    
    # Training XGBRegressor. Unlike svdd and glosh xgbr performs well with balanced data sets 
    def xgbrl(self):
        self.lab = np.concatenate((np.ones((self.n_of_good, )), np.zeros((self.n_of_bad, ))), axis=0)
        data_dmatrix = xgb.DMatrix(data=self.all_features,label=self.lab)
        
        
        # Parameter dictionary specifying base learner
        param = {"booster":"gblinear", "objective":"reg:linear"}
         
        self.xgb_r = xgb.train(params = param, dtrain = data_dmatrix, num_boost_round = 10)
    
    
    
    
# Simple SVDD with multiple feture sets as inputs

# Instance with training and test data sets as inputs
a1 = Anomdet()

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[1,1,1,0,1])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[1,1,1,0,1])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata=[]
g1 = len(os.listdir("fin_data/goodt/"))
b1 = len(os.listdir("fin_data/badt/"))
goodtdata = a1.getdata("fin_data/goodt/",[1,1,1,0,1])
testdata = testdata.append(goodtdata)
badtdata = a1.getdata("fin_data/badt/",[1,1,1,0,1])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

# Predicting with the test features
temp = []
temp = a1.osvm_model.predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred1 = [0 if i==-1 else 1 for i in temp]

# actual labels for the images 
truth = np.concatenate((np.ones((g1, )), np.zeros((b1, ))), axis=0)

# Performance
print('Accuracy:',accuracy_score(pred1,truth))
print('Confusion matrix:\n',confusion_matrix(pred1,truth))

# to predict over a dataset y

preddata = a1.getdata("fin_data/trial/",[1,1,1,0,1])
imagepred_df = pd.DataFrame(preddata)
imagepred_df.drop(0,axis=1,inplace=True)
allpred_features = np.array(imagepred_df)

# Predicting with the test features
predicty = []
predicty = a1.osvm_model.predict(allpred_features)
# Since outliers are labeled as -1 in svdd and glosh
pred1 = [0 if i==-1 else 1 for i in predicty]




# Simple GLOSH with multiple feture sets as inputs

# Instance with training and test data sets as inputs
a2 = Anomdet()

# Feature selector given in the form of an array
traindata=[]
gooddata = a2.getdata("fin_data/good/",[1,1,1,0,1])
traindata = traindata.append(gooddata)
baddata = a2.getdata("fin_data/bad/",[1,1,1,0,1])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an glosh with the given training data 
a2.glosh(alltrain_features)

testdata=[]
g1 = len(os.listdir("fin_data/goodt/"))
b1 = len(os.listdir("fin_data/badt/"))
goodtdata = a2.getdata("fin_data/goodt/",[1,1,1,0,1])
testdata = testdata.append(goodtdata)
badtdata = a2.getdata("fin_data/badt/",[1,1,1,0,1])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)


# Predicting with the test features
temp = []
temp = a2.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred1 = [0 if i==-1 else 1 for i in temp]

# actual labels for the images 
truth = np.concatenate((np.ones((g1, )), np.zeros((b1, ))), axis=0)

# Performance
print('Accuracy:',accuracy_score(pred1,truth))
print('Confusion matrix:\n',confusion_matrix(pred1,truth))






# Individual feature extraction for ensmeble models
a1 = Anomdet()

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[1,0,0,0,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[1,0,0,0,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata=[]
g1 = len(os.listdir("fin_data/goodt/"))
b1 = len(os.listdir("fin_data/badt/"))
goodtdata = a1.getdata("fin_data/goodt/",[1,0,0,0,0])
testdata = testdata.append(goodtdata)
badtdata = a1.getdata("fin_data/badt/",[1,0,0,0,0])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred1 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred6 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,1,0,0,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,1,0,0,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

goodtdata = a1.getdata("fin_data/goodt/",[0,1,0,0,0])
testdata = testdata.append(goodtdata)
badtdata = a1.getdata("fin_data/badt/",[0,1,0,0,0])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred2 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred7 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,0,1,0,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,0,1,0,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

goodtdata = a1.getdata("fin_data/goodt/",[0,0,1,0,0])
testdata = testdata.append(goodtdata)
badtdata = a1.getdata("fin_data/badt/",[0,0,1,0,0])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred3 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred8 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,0,0,1,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,0,0,1,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

goodtdata = a1.getdata("fin_data/goodt/",[0,0,0,1,0])
testdata = testdata.append(goodtdata)
badtdata = a1.getdata("fin_data/badt/",[0,0,0,1,0])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred4 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred9 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,0,0,0,1])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,0,0,0,1])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

goodtdata = a1.getdata("fin_data/goodt/",[0,0,0,0,1])
testdata = testdata.append(goodtdata)
badtdata = a1.getdata("fin_data/badt/",[0,0,0,0,1])
testdata = testdata.append(badtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)


temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred5 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred10 = [0 if i==-1 else 1 for i in temp]

# actual labels for the images 
truth = np.concatenate((np.ones((g1, )), np.zeros((b1, ))), axis=0)

# Performance
print('Accuracy:',accuracy_score(pred1,truth))
print('Confusion matrix:\n',confusion_matrix(pred1,truth))




# generating training data dor ensemble
ensfeat = []
for i in range(a1.total_test_num):
    temp = [pred1[i],pred2[i],pred3[i],pred4[i],pred5[i],pred6[i],pred7[i],pred8[i],pred9[i],pred10[i]]
    ensfeat.append(temp)
    
    
    
    
    
    
    # The ensemble takes outputs from individual classifiers and their respective labels  as inputs 

class EnsmebleMod:
    def __init__(self, preds, truths):
        self.preds = preds
        self.truths = truths
    
    # Multilayer perceptron as an ensemble
    def mlpc(self):
        X_train, X_test, y_train, y_test = train_test_split(self.preds, self.truths, test_size=0.2, random_state=3)
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        self.clf.fit(X_train,y_train)
        finpreds = self.clf.predict(X_test)


        print('Accuracy:',accuracy_score(finpreds,y_test))
        print('Confusion matrix:\n',confusion_matrix(finpreds,y_test))
        
    
    # xgboost as an ensemble
    def xgbr(self):
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(self.preds, self.truths, test_size=0.2, random_state=123)
        # Train and test set are converted to DMatrix objects,
        # as it is required by learning API.
        train_dmatrix = xgb.DMatrix(data = X_train, label = y_train)
        test_dmatrix = xgb.DMatrix(data = X_test, label = y_test)
        
        # Parameter dictionary specifying base learner
        param = {"booster":"gblinear", "objective":"reg:linear"}
         
        self.xgb_r = xgb.train(params = param, dtrain = train_dmatrix, num_boost_round = 10)
        xgpred = self.xgb_r.predict(test_dmatrix)

        print('Accuracy:',accuracy_score(xgpred,y_test))
        print('Confusion matrix:\n',confusion_matrix(xgpred,y_test))
        
    # Bagging    
    def bagg(self):
        X_train, X_test, y_train, y_test = train_test_split(self.preds, self.truths, test_size=0.2, random_state=123)
        base_cls = DecisionTreeClassifier()
        self.model = BaggingClassifier(base_estimator = base_cls, n_estimators = 500, random_state = 8)
        self.model.fit(X_train,y_train)
        
        
        
        
        
        
        #Creating an instance of EnsembleMod
e1 = EnsmebleMod(ensfeat, truth)

# NN ensemble
e1.mlpc()




# Individual feature extraction for ensmeble models
a3 = Anomdet()

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[1,0,0,0,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[1,0,0,0,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata=[]
g1 = len(os.listdir(Y))

testdata = a1.getdata(Y,[1,0,0,0,0])


imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred1 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred6 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,1,0,0,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,1,0,0,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata = a1.getdata(Y,[0,1,0,0,0])
testdata = testdata.append(goodtdata)

imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred2 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred7 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,0,1,0,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,0,1,0,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata = a1.getdata(Y,[0,0,1,0,0])


imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred3 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred8 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,0,0,1,0])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,0,0,1,0])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata = a1.getdata(Y,[0,0,0,1,0])


imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)

temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred4 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred9 = [0 if i==-1 else 1 for i in temp]

# Feature selector given in the form of an array
traindata=[]
gooddata = a1.getdata("fin_data/good/",[0,0,0,0,1])
traindata = traindata.append(gooddata)
baddata = a1.getdata("fin_data/bad/",[0,0,0,0,1])
traindata = traindata.append(baddata)

image_df = pd.DataFrame(traindata)
image_df.drop(0,axis=1,inplace=True)
alltrain_features = np.array(image_df)

# Fitting an svdd with the given training data 
a1.svdtrain(alltrain_features)

testdata = a1.getdata(Y,[0,0,0,0,1])



imaget_df = pd.DataFrame(testdata)
imaget_df.drop(0,axis=1,inplace=True)
alltest_features = np.array(imaget_df)


temp = []
temp = a1.svm_model.predict(a1.alltest_features)

pred5 = [0 if i==-1 else 1 for i in temp]

# Fitting an glosh with the given training data 
a1.glosh(alltrain_features)
temp = []
temp = a1.clusters.fit_predict(alltest_features)
# Since outliers are labeled as -1 in svdd and glosh
pred10 = [0 if i==-1 else 1 for i in temp]

ensfeat = []
for i in range(a1.total_test_num):
    temp = [pred1[i],pred2[i],pred3[i],pred4[i],pred5[i],pred6[i],pred7[i],pred8[i],pred9[i],pred10[i]]
    ensfeat.append(temp)

# Performance
print('Accuracy:',accuracy_score(pred1,truth))
print('Confusion matrix:\n',confusion_matrix(pred1,truth))

# To predict anomalous images in a dataset Y
prediction = e1.clf.predict(ensfeat)




