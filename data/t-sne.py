# Performing t-sne for our given dataset

import time
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import scale
import pickle
from matplotlib import pyplot as plt

def loadData(path):

	df = pd.read_csv(path)
	print(df.shape)
	y=df[df.columns[-1]].tolist()
	df=df.drop(df.columns[-1],axis=1)
	names=df[df.columns[0]]
	df=df.drop(df.columns[0],axis=1)
	X=scale(df.values).tolist()

	print(type(X), type(y))
	return X, y


def t_sne(X, y, n):

	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=800)
	x_embedded = tsne.fit_transform(X[:n])

	print("Done. Time elapsed = ", time.time()-time_start)
	#print(x_embedded)
	print("Final embeddings : ", type(x_embedded), x_embedded.shape)
	dumpFile=open("tsne.pickle","wb")
	pickle.dump(x_embedded,dumpFile)

	return x_embedded
	
def plotEmbeddings(x_embedded, y, n):

	x0 = []
	y0 = []
	x1 = []
	y1 = []

	# Red is label 0
	# Blue is label 1 
	for i in range(n):
		if y[i] == 0:
			x0.append(x_embedded[i][0])
			y0.append(x_embedded[i][1])
		else :
			x1.append(x_embedded[i][0])
			y1.append(x_embedded[i][1])

	plt.scatter(x0, y0, color='red')
	plt.scatter(x1, y1, color='blue')
	plt.show()



if __name__ == '__main__' :

	path = 'final_training_data.csv'
	X, y = loadData(path)
	n_samples=3000
	try:
		with open("tsne.pickle","rb") as f:
			x_embedded = pickle.load(f)
	except:
		x_embedded = t_sne(X, y, n_samples)
	
	plotEmbeddings(x_embedded, y, n_samples)