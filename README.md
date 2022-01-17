# car-recommendation-
#importing Library 
import numpy as np
import pandas as pd

#importing dataset 
data=pd.read_csv("car_dataset.csv")

data.describe()

data.head()

data.head()['Name'].values

data.shape

#selecting features from data 
data= data[['Name','EMI','Price','FUEL TYPE','Price range','Seating Capacity']]

data.isnull().sum()

data.dropna(inplace=True)

data.duplicated().sum()

data.iloc[100].Price

data.iloc[0].Name

data['Price range'][0]

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=200)

vectors=cv.fit_transform(data['Price range']).toarray()

vectors[0]

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

similarity[0]

sorted(similarity[0],reverse=True)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:5]

def recommend(car):
   car_id=data[data['Price']==car].index[0]
   distance=similarity[car_id]
   car_list=sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:10]

   for i in car_list:
      print(data.iloc[i[0]].Name)


recommend('1.44 Crore')

import pickle

pickle.dump(data,open('Price.pkl','wb'))

data['Price'].values[1:10]

pickle.dump(data.to_dict(),open('car_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))




