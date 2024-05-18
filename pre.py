#import essential libraries
import pandas as pd
import numpy as np
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


#read dataset using pandas
data = pd.read_csv("Project_Dataset/dataset.csv")
#printing rows
print(data.head())
print(data.shape)
#getting columns
print(data.columns)

#Perform preprocessing
print(data['Label'].value_counts())

#checking missing values
print(data.isna().sum())

#removing missing values
data.dropna(inplace=True)

print(data.isna().sum())

print(data)

#Labels
# 0 - Easy ( Low )
# 1 - Medium ( Medium )
# 2 - Hard ( High )

def get_preprocess1(data):
    #creating an empty list
    non_features = []

    # store number of columns
    number_of_columns = data.shape[1] 
    print(number_of_columns)


    for i in range(number_of_columns):
        if len(data.iloc[:,i].unique()) == 1:

            #perform appending
            non_features.append(data.columns[i])

    # print(non_features)
    # print(len(non_features))

    return non_features


#function call
non_features=get_preprocess1(data)

print(non_features)
print(len(non_features))

new_df = data.copy()
# delete non-informative features 
new_df = data.drop(columns=non_features) 

print(new_df.columns)


#Anova Method
def select_features(x, y):

    fs = SelectKBest(score_func=f_classif, k="all")

    # learn relationship from training data
    fs.fit(x, y)

    return fs


#data division (independent variable & dependent variable)
y=new_df['Label']
x=new_df.drop(['Label'],axis=1)


# feature selection(function call)
fs = select_features(x, y)


column_names=[]
# iterating the columns
j=0
for col in x.columns:
    print(col)
    print(j)
    column_names.append(col)
    j=j+1

print(column_names)
print(len(column_names))

feature_list=[]
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
	feature_list.append(fs.scores_[i])

print(feature_list)
print(len(feature_list))

dictionary = dict(zip(column_names, feature_list))
clean_dict = {k: dictionary[k] for k in dictionary if not pd.isna(dictionary[k])}
sorted_d = dict( sorted(clean_dict.items(), key=operator.itemgetter(1),reverse=True))

print(sorted_d)
print(len(sorted_d))


