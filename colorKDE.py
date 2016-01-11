import numpy as np
import pickle
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

class colorKDE(object):
	def __init__(self,data=np.array([])):
		self.data = data
	
		
	def runKDE(self,bandwidth=0.2,use_opt=False):
		'''
		Generate the KDE and run with given bandwith
		
		If use_opt is specified, ruCVSearch must have been run already
		'''
		if use_opt:
			self.kde = KernelDensity(bandwidth=self.optimal_bandwidth)
		else:
			self.kde = KernelDensity(bandwidth=bandwidth)
		
		self.kde.fit(self.data)
		
	def runCVSearch(self,search_range=np.linspace(0.01,1.0,50),folds=20):
		self.grid = GridSearchCV(KernelDensity(),{'bandwidth':search_range},\
			cv=folds)
		self.grid.fit(self.data)
		self.optimal_bandwidth=self.grid.best_params_['bandwidth']
		print 'Optimal bandwidth: ' + str(self.optimal_bandwidth)
		
	def score_samples(self,x):
		'''
		Replicate score_samples functionality so both saves
		can be treated the same
		'''
		return self.kde.score_samples(x)
		
	def sample(self,n_samples):
		'''
		Replicate samples functionality so both saves
		can be treated the same
		'''
		return self.kde.sample(n_samples=n_samples)
		
	
	def save(self,filename,full=True):
		'''
		Save current state of the object
		
		If full is false, only save self.kde
		'''
		if full:
			#save the entire object, including data
			pickle.dump(self,open(filename,'wb'),protocol=-1)
			
		else:
			#only save the .kde object
			pickle.dump(self.kde,open(filename,'wb'),protocol=-1)