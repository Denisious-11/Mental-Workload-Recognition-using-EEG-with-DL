#import essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts1
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization,GRU,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns



final_df=pd.read_csv("Project_Dataset/final_dataset_preprocessed.csv")

#data division
y_final = final_df['Label']
x_final = final_df.drop(['Label'], axis=1)



# TRAIN - TEST SPLITTING
x_train, x_test, y_train, y_test = tts1(x_final, y_final, test_size=0.2)
print("\nTraining set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting set")
print(x_test.shape)
print(y_test.shape)

print(x_train)


#Data balancing using SMOTE
counter = Counter(y_train)
print("__________________BEFORE::::::", counter)

smt = SMOTE(k_neighbors=1)

x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)

counter = Counter(y_train_sm)
print("___________________AFTER:::::::", counter)


print("x_train_sm shape:", x_train_sm.shape)
print("y_train_sm shape:", y_train_sm.shape)

#standardization
scaler = StandardScaler()
x_train_sm = scaler.fit_transform(x_train_sm)
x_test = scaler.transform(x_test)
pickle.dump(scaler,open('Project_Extra/scaler_dl1.pkl','wb'))


#Applying dimension expansion
x_train_sm = np.expand_dims(x_train_sm, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print(x_train_sm.shape)
print(y_train_sm.shape)
print(x_test.shape)
print(y_test.shape)

def model_cnn(x_train):
	
    _build_model=Sequential()

    _build_model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
    _build_model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    _build_model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    _build_model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    _build_model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    _build_model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    _build_model.add(Dropout(0.2))

    _build_model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
    _build_model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    _build_model.add(Flatten())
    _build_model.add(Dense(units=32, activation='relu'))
    _build_model.add(Dropout(0.3))

    _build_model.add(Dense(units=3, activation='softmax'))

    #compiling the _build_model
    _build_model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

    #_build_model summary
    print(_build_model.summary())
    return _build_model

_build_model=model_cnn(x_train_sm)

#saving the _build_model(checkpoint)
checkpoint=ModelCheckpoint("Project_Saved_Models/Trained_CNN_model1.h5",monitor="accuracy",save_best_only=True,verbose=1)#when training deep learning _build_model,checkpoint is "WEIGHT OF THE _build_model"
#Training
history=_build_model.fit(x_train_sm, y_train_sm, batch_size=16, epochs=300, validation_data=(x_test, y_test), callbacks=[checkpoint])



#plot accuracy and loss 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('_build_model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('Project_Extra/acc_plot1.png')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('_build_model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('Project_Extra/loss_plot1.png')
plt.show()

from sklearn.metrics import classification_report,confusion_matrix

y_pred_probs = _build_model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test, y_pred))


# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('Project_Extra/confusion_matrix.png')
plt.show()