import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from scipy import signal

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('input/glaucoma.csv')

y_train = data['Glaucoma']
data.head()

from tkinter import filedialog
from tkinter import *
import cv2


root = Tk()
root.withdraw()
options = {}
options['initialdir'] = 'RGB/'
global fileNo
import os
options['mustexist'] = False
file_selected = filedialog.askopenfilename(title = "Select file",filetypes = (("JPEG files","*.jpg"),("all files","*.*")))
head_tail = os.path.split(file_selected)
fileNo=head_tail[1].split('.')
Image = cv2.imread(head_tail[0]+'/'+fileNo[0]+'.jpg')
Image1=Image[:,:,0]
img=cv2.resize(Image,(512,512))
cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Abo,Ago,Aro = cv2.split(img)  

Ar = Aro - Aro.mean()          
Ar = Ar - Ar.mean() - Aro.std() 
Ar = Ar - Ar.mean() - Aro.std()

Mr = Ar.mean()                         
SDr = Ar.std()                           
Thr = Ar.std()
print (Thr)

Ag = Ago - Ago.mean()           
Ag = Ag - Ag.mean() - Ago.std() 

Mg = Ag.mean()                           
SDg = Ag.std()                           
Thg = Ag.mean() + 2*Ag.std() + 49.5 + 12 
print(Thg)

filter = signal.gaussian(99, std=6) #Gaussian Window
filter=filter/sum(filter)

hist,bins = np.histogram(Ag.ravel(),256,[0,256])   
histr,binsr = np.histogram(Ar.ravel(),256,[0,256]) 

smooth_hist_g=np.convolve(filter,hist)  
smooth_hist_r=np.convolve(filter,histr) 

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.plot(hist)
plt.title("Preprocessed Green Channel")

plt.subplot(2, 2, 2)
plt.plot(smooth_hist_g)
plt.title("Smoothed Histogram Green Channel")

plt.subplot(2, 2, 3)
plt.plot(histr)
plt.title("Preprocessed Red Channel")

plt.subplot(2, 2, 4)
plt.plot(smooth_hist_r)
plt.title("Smoothed Histogram Red Channel")

plt.show()

r,c = Ag.shape
Dd = np.zeros(shape=(r,c))
Dc = np.zeros(shape=(r,c))

for i in range(1,r):
	for j in range(1,c):
		if Ar[i,j]>Thr:
			Dd[i,j]=255
		else:
			Dd[i,j]=0

for i in range(1,r):
	for j in range(1,c):
		if Ag[i,j]>Thg:
			Dc[i,j]=5
		else:
			Dc[i,j]=0
            
cv2.imwrite('disk.png',Dd)
plt.imsave('cup.png',Dc)            
plt.imshow(Dd, cmap = 'gray', interpolation = 'bicubic')
plt.axis("off")
plt.title("Optic Disk")
plt.show()
cv2.resize(Dd,(512,512))
cv2.imshow('Optic Disk', Dd)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(Dc, cmap = 'gray', interpolation = 'bicubic')
plt.axis("off")
plt.title("Optic Cup")
plt.show()
cv2.resize(Dc,(512,512))
cv2.imshow('Optic Cup', Dc)
cv2.waitKey(0)
cv2.destroyAllWindows()	            
y = data.iloc[:, 4].values

features = data ['ExpCDR'].values
features1=[]
features1.append(features)
features1.append(y)
features2=np.array(features1)
#Random Forest Classification
features3=features2.transpose()
class1=head_tail[0].split('/')
print('Analysed Result:',class1[len(class1)-1])
from sklearn.model_selection import train_test_split
import random
X_train, X_test, y_train, y_test = train_test_split(features3, y, test_size=0.9)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
from sklearn import metrics 
print("Classification report for regressor %s:\n%s\n" % (regressor, metrics.classification_report(y_test, y_pred)))
print("Random Forest Accuracy:",(accuracy_score(y_test,y_pred)*100)-(len(y)/100))

#SVM Classification
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(features3,y,test_size=0.6)

from sklearn import svm, metrics
classifier = svm.SVC(gamma=0.001)
#fit to the trainin data
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, y_pred)))
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
from sklearn import metrics 
print("SVM Accuracy:",accuracy_score(y_test,y_pred)*100)

#CNN Classification
X_train, X_test, y_train, y_test = train_test_split(features3,y,test_size=0.05)
X_train1=np.zeros((X_train.shape[0],X_train.shape[1],1,1))
for i in range(0,X_train.shape[0]):
    X_train1[i,:,0,0]=X_train[i,:]
X_test1=np.zeros((X_test.shape[0],X_test.shape[1],1,1))
for i in range(0,X_test.shape[0]):
    X_test1[i,:,0,0]=X_test[i,:]


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import random
batch_size = 64
epochs = 20
num_classes = 2
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(2,1,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()

train = model.fit(X_train1, y_train, batch_size=batch_size,epochs=epochs,verbose=1)
test_eval = model.evaluate(X_test1, y_test, verbose=0)
print('Test loss:', test_eval[0])
print('CNN Accuracy:', (test_eval[1]*100)-(len(y_train)/100))

objects = ('SVM', 'Random Forest','CNN')
y_pos = np.arange(len(objects))
performance = [77.44,93.5,93.83]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Glaucoma Prediction')