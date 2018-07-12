import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

'''
We are going to calculate a matrix that summarizes how our variables all relate to one another.

We then break this matrix down into two separate components: direction and magnitude. 
We can then understand the directions of our data and its magnitude

PCA finds a new set of dimensions (or a set of basis of views) such that all the dimensions are orthogonal (and hence linearly independent) and ranked according to the variance of data along them. It means more important principle axis occurs first. (more important = more variance/more spread out data).

'''

data=pd.read_csv("trainingDataCleaned.csv",low_memory=False)

#filling NA Values
for column in list(data.columns.values):
	data[column].fillna(data[column].mean())

#taking only the values
x=data.values

#scaling the values
x=scale(x)

#taking all the attributes
pca=PCA(n_components=3803)
pca.fit(x)

#getting the variance
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()