from sklearn.neural_network import MLPClassifier
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import classification_report, confusion_matrix  
from keras.layers.core import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import regularizers
from sklearn import metrics
from keras.layers import BatchNormalization



path = "."
#path = "/Users/willemvandemierop/Documents/Master AI/Introduction to AI/Coursework"
filename_read = os.path.join(path, "All_data_patients_correct.csv")
patient_all = pd.read_csv(filename_read)
PAL = patient_all
print("All patients\n", PAL.head())
PAL = shuffle(PAL)
print("All patients shuffled\n",PAL.head())
print("Patients data size", PAL.shape)

min_max_scaler = preprocessing.MinMaxScaler()
PAL[['heartbeat']] = min_max_scaler.fit_transform(PAL[['heartbeat']].values)

print("\nPatients data normalized heartbeat\n", PAL.head())

result = []
for x in PAL.columns:
    if x != 'label':
        result.append(x)

X = PAL[result].values
X = np.delete(X,0,1)
y = PAL['label'].values
print("X data \n", X[0:5])

print("Y data \n", y[0:5])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.25, random_state = 42)
#smt = SMOTE()
#X_train, y_train = smt.fit_sample(X_train, y_train)

one_hot = np.identity(7)
train_labels_one_hot = []
test_labels_one_hot = []
val_labels_one_hot = []
for i in range(y_train.shape[0]):
    train_labels_one_hot.append(one_hot[y_train[i]])   

for i in range(y_test.shape[0]):
    test_labels_one_hot.append(one_hot[y_test[i]]) 
    
for i in range(y_val.shape[0]):
    val_labels_one_hot.append(one_hot[y_val[i]])   

print(X_train.shape[1])

#X_train_reshaped = X_train.reshape(32*32*3, 73257)
#X_test_reshaped = X_test.reshape(32*32*3, 26032)
input_shape = X_train.shape[1]
print('input_shape', input_shape)
print("x train shape", X_train.shape)

print("\nInitializing Keras\n")
model = Sequential()

model.add(Dense(100, input_shape = (input_shape,),  activation = 'relu'))
model.add(Dropout(0.04))
model.add(Dense(50, activation = 'relu'))#,kernel_regularizer = regularizers.l2(0.01)))
model.add(Dropout(0.04))
model.add(Dense(25, activation= 'relu'))#, kernel_regularizer = regularizers.l2(0.01)))
# model.add(Dropout(0.04))
model.add(Dense(7,activation = 'softmax'))#,kernel_regularizer = regularizers.l2(0.01)))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam') # adaptive momentum solves issue of local minimum
#model.summary()

monitor = EarlyStopping(monitor='loss', min_delta= 1e-3, patience = 5, verbose = 1, mode = 'auto' )

#model.fit(X_train, np.array(train_labels_one_hot),callbacks = [monitor], verbose = 2, epochs = 25)
history = model.fit(X_train, np.array(train_labels_one_hot), verbose = 2, epochs = 1000,validation_data=(X_val, np.array(val_labels_one_hot)))
pred = model.predict(X_train)
score = np.sqrt(metrics.mean_squared_error(pred,train_labels_one_hot))
print(f"Final score (RMSE): {score}")

print(pred.shape)
print(y_train.shape)
corrects,wrongs = 0,0
for i in range(len(pred)):
    res = pred[i]
    res_max = res.argmax()
    if res_max == y_train[i]:
        corrects += 1
    else:
        wrongs += 1
        
print("Accuracy train: ", corrects / (corrects + wrongs))

pred = model.predict(X_test)
score = np.sqrt(metrics.mean_squared_error(pred,test_labels_one_hot))
print(f"Final Test score (RMSE): {score}")

corrects,wrongs = 0,0
for i in range(len(pred)):
    res = pred[i]
    res_max = res.argmax()
    if res_max == y_test[i]:
        corrects += 1
    else:
        wrongs += 1
        
print("Accuracy test: ", corrects / (corrects + wrongs))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

#plt.savefig('overfitting with drop out 0.04 in 2 layers, NN100.png')
plt.show()