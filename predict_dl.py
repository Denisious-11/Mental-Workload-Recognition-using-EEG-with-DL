#import necessary libraries
import pandas as pd
import joblib
import numpy as np
import pickle
from tensorflow.keras.models import load_model


#loading the saved model
loaded_model=load_model("Project_Saved_Models/Trained_CNN_model.h5")
#loading standardscaler
scaler=pickle.load(open('Project_Extra/scaler_dl.pkl','rb'))

info=[]
parameters=['Wavelet Approximate Entropy','Variance of Vertex to Vertex Slope','Wavelet Detailed Energy',
			'Hjorth_Activity','FFT Beta MaxPower','1st Difference Max','FFT Alpha MaxPower',
			'Wavelet Approximate Std Deviation','FFT Theta MaxPower','Coeffiecient of Variation']


Wavelet_Approximate_Entropy=input("Wavelet_Approximate_Entropy : ")
info.append(Wavelet_Approximate_Entropy)
Variance_of_Vertex_to_Vertex_Slope=input("Variance_of_Vertex_to_Vertex_Slope : ")
info.append(Variance_of_Vertex_to_Vertex_Slope)
Wavelet_Detailed_Energy=input("Wavelet_Detailed_Energy : ")
info.append(Wavelet_Detailed_Energy)
Hjorth_Activity=input("Hjorth_Activity : ")
info.append(Hjorth_Activity)
FFT_Beta_MaxPower=input("FFT_Beta_MaxPower : ")
info.append(FFT_Beta_MaxPower)
First_Difference_Max=input("1st_Difference_Max : ")
info.append(First_Difference_Max)
FFT_Alpha_MaxPower=input("FFT_Alpha_MaxPower : ")
info.append(FFT_Alpha_MaxPower)
Wavelet_Approximate_Std_Deviation=input("Wavelet_Approximate_Std_Deviation : ")
info.append(Wavelet_Approximate_Std_Deviation)
FFT_Theta_MaxPower=input("FFT_Theta_MaxPower : ")
info.append(FFT_Theta_MaxPower)
Coeffiecient_of_Variation=input("Coeffiecient_of_Variation : ")
info.append(Coeffiecient_of_Variation)

my_dict=dict(zip(parameters,info))

#convert dict into dataframe
my_data=pd.DataFrame(my_dict,index=[0])

feat=np.array(my_data)
# print(feat)
#perform standardization
feat=scaler.transform(feat)
# print(feat)
#expand dimension
feat = np.expand_dims(feat, axis=2)
# print(feat)
# print(feat.shape)

# dataframe is putting into the MODEL to make PREDICTION
my_pred = loaded_model.predict(feat)
print(my_pred)
my_pred=np.argmax(my_pred)
print(my_pred)

print("\n*************Result**************")

if my_pred==0:
	print("Mental Cognitive Workload : Low ")

if my_pred==1:
	print("Mental Cognitive Workload : Medium ")

if my_pred==2:
	print("Mental Cognitive Workload : High ")

