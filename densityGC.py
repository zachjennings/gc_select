import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import scipy.misc

#Reasonable covariance and means for gi and ri colors, measured from N3115 spectroscopically-confirmed GCs
default_cov_matrix = np.array([[0.03574185,  0.01381638],[ 0.01381638,  0.00612115]])
default_cov_matrix_bim = np.array([[[0.0041778367357417747, 0.0018160584767313343],[0.0018160584767313343, 0.0011661958108050926]],\
	[[0.023774259439179244, 0.0073008128420189748],[0.0073008128420189748, 0.0030141805937676275]]])
default_means_bim = np.array([[1.0896732866320709,0.32906797455385733],[0.80145778134844314,0.20049467773114149]])
default_fractions = np.array([0.5,0.5])
default_lum_means = np.array([23.])
default_lum_sigs = np.array([1.4])

class densityGC(object):
	'''
	A class to hold the probabilty density model for the GCs.
	
	Parameters
	----------
	bimodality: boolean, optional
		whether to fit for two modes or just one
	ellipticity: boolean, optional
		whether to flatten the GC distribution or not
	radial_profile: string, optional
		the type of radial profile to look for. Currently
		only takes 'exponential', will implement 'sersic'
		and 'pareto' in future
	fixed_cov: boolean, optional
		whether to fix the covariance matrix when doing fitting
	cov: 3-dimensional array, optional
		covariance matrix used to generate mock data,
		and perform fitting if fixed_cov is true
		must be a i x j x j array, where i = number of modes,
		j=number of bands minus one
	means: i x j x 1 array
		used to generate mock data, and perform fitting if
		fixed_cov is true. i = number of modes, j=number of bands-1
	fractions: i x 1 array
		relative numbers of sources of each of i modes
	'''
	
	def __init__(self,bimodality=False,spatial_bimodality=False,radial_profile='exponential',\
		ellipticity=False,fixed_lum=True,lum=False,\
		fixed_cov=True,cov=default_cov_matrix_bim,means=default_means_bim,spatial=np.array([]),\
		lum_mean=default_lum_means,lum_sig=default_lum_sigs,center=[0.,0.]):
			
		#specify the covariance matrix for mock data and fitting
		self.cov=cov
		self.means = means
		self.spatial = spatial
		self.lum_means = lum_mean
		self.lum_sigs = lum_sig
		self.center = center
		
		#determine the number of required parameters
		self.ellipticity=ellipticity
		self.radial_profile = radial_profile
		self.lum=lum
		
		#check if the user wants to fix the covariance matrix
		self.fixed_cov = fixed_cov
		
		#check if the user wants to consider multimodal data
		self.bimodality = bimodality
		
		#check if user wants to fix luminosity function parameters
		self.fixed_lum = fixed_lum
	
	def lnLikeMag(self,mag,lum_means=np.array([]),lum_sigs=np.array([])):
		'''
		Return the log of the Gaussian Likelihood Distribution. The distribution called will
		be the same for both the color distribution and the luminosity function.
		'''
		
		if self.fixed_lum:
			lum_means = self.lum_means
			lum_sigs = self.lum_sigs
		else:
			lum_means = lum_means
			lum_sigs = lum_sigs
			
		like_shape = (lum_means.size,mag.size)
		ln_like = np.zeros(like_shape)
		
		for i in np.arange(lum_means.size):	
			ln_like[i,:] = stats.norm.logpdf(mag,loc=lum_means[i],scale=lum_sigs[i])
			
		return scipy.misc.logsumexp(ln_like,axis=0)
	
	def lnLikeColor(self,color,cov=np.array([]),means=np.array([]),fractions=[1.]):
		'''
		Return the log of i multivariate Gaussians 
		
		Parameters
		----------
		color = j x n array
			j = number of bands -1, n = number of points
		covariance: array-like, optional
			if given, the i x j x j covariance matrix
		means: array-like, optional
			if given, the i x j x j covariance matrix
		frac: array-like, optional
			i x 1 array containing the relative fractions in the mixture model
			
		Returns: 1d array
			returns the summed color likelihood for every data point
		'''
		if self.fixed_cov:
			cov = self.cov
			means = self.means

		else:	
			cov = cov
			means = means

			
		self.color = color
		self.fractions = fractions
		
		like_shape = (fractions.size,color[:,0].size)
		self.ln_like = np.zeros(like_shape)
		
		for i in np.arange(fractions.size):
			self.ln_like[i,:] = np.log(fractions[i]) + stats.multivariate_normal.logpdf(color,mean=means[i],cov=cov[i]) 
		
		return scipy.misc.logsumexp(self.ln_like,axis=0)
		
		#return -.5 * np.log(np.linalg.det(cov)) - \
		#		.5 * (col - mean)*np.linalg.inv(cov)*(col - mean).transpose() - \
		#		self.n_colors/2. * np.log(2.*np.pi)
				
	def lnLikeSpatialExp(self,positions,spatial):
		'''
		Return the log of the GC Spatial Distribution, assuming exponential
		'''
		r_s = spatial[0]
		if self.ellipticity:
			q = spatial[1]
			pa = spatial[2]
		else:
			q = 1.0
			pa = 0.0
			
		x_prime = positions[:,0]
		y_prime = positions[:,1]
		
		x_prime = x_prime - self.center[0]
		y_prime = y_prime - self.center[1]
		
		x = np.sin(pa) * y_prime + np.cos(pa) * x_prime
		y = np.cos(pa) * y_prime - np.sin(pa) * x_prime		
		y = y / q
		
		dist = y**2 + x**2
		return stats.expon.logpdf(dist,scale=r_s)  - np.log(2 * np.pi * q)
		
	def lnLikeSpatialPareto(self,x,y,r_s,x_cen=0,y_cen=0,q=1,pa=0.):
		'''
		Return the log of the GC Spatial Distribution, assuming exponential
		'''
		dist = (x-x_cen)**2 + (y-y_cen)**2
		return stats.pareto.logpdf(dist,scale=r_s,loc=-1)# - np.log(2 * np.pi)
		
	def lnLike(self,data,fractions=[1.0],means=[1.0],cov=[1.0],spatial=[1.0],\
		lum_means=[1.0],lum_sigs=[1.0]):
		'''
		Return the log likelihood for the full sample

		Parameters
		----------
		data = tuple
			(color array, spatial array)
		fractions: array-like, optional
			i x 1 array containing the relative fractions in the mixture model
		covariance: array-like, optional
			if given, the i x j x j covariance matrix
		means: array-like, optional
			if given, the i x j x j covariance matrix
		spatial:array-like, optional
			parameters (scale, ellipticity, and PA) for spatial distributions
			can be i x 3, where i = number of modes, or 1 x 3

		Returns: 1d array
			returns the summed color and spatial likelihood for every data point
		'''
		color,mags,positions = data
		col_like = self.lnLikeColor(color,cov,means,fractions=fractions)
		self.col_like = col_like
		
		if self.lum:
			self.lum_like = self.lnLikeMag(mags,lum_means=lum_means,lum_sigs=lum_sigs)
			col_like = col_like + self.lum_like

		if self.radial_profile == 'exponential':
			return col_like + self.lnLikeSpatialExp(positions,spatial)

		elif self.radial_profile == 'pareto':
			return col_like + self.lnLikeSpatialPareto(positions,spatial)

		else:
			return col_like
		
	
	def genSpatial(self,n_points=[200],plot=False):
		'''
		Draw the x and y of n gcs distributed according to ellipse with major axis a,
		minor/major ratio q, position angle theta
		'''
		r_s = self.spatial[0]
		if self.ellipticity:
			q = self.spatial[1]
			pa = self.spatial[2]
		else:
			q = 1.0
			pa = 0.0
		
		if self.radial_profile == 'exponential':
			rho = np.sqrt(stats.expon.rvs(scale=r_s,size=n_points))
			
		elif self.radial_profile == 'pareto':
			rho = np.sqrt(stats.pareto.rvs(scale=r_s,loc=-1,size=n_points))
		
		else:
			rho = np.sqrt(np.random.uniform(0,r_s,n_points))
		
		phi	= np.random.uniform(0.0,2.0*np.pi,size=n_points)
		x = rho * np.cos(phi)
		y = rho * np.sin(phi)
		x = x 
		y = y * q 
		x_prime = np.cos(pa)*x - np.sin(pa)*y 
		y_prime = np.sin(pa)*x + np.cos(pa)*y
		
		x_prime = x_prime + self.center[0]
		y_prime = y_prime + self.center[1]
		
		if plot:
			plt.scatter(x_prime,y_prime,s=0.5,marker='.',color='b',alpha=0.5)
			plt.show()
		
		return np.array([x_prime,y_prime]).T
		
	def genColors(self,n_points=[100,100],plot=False):
		'''
		Draw the gi and ri colors of n GCs distributed according to i MVNs
		
		n_points: array-like, optional
			i x 1 array, containing number of GCs to create for each mode
		'''
		cumu_points = np.cumsum(n_points)
		colors = np.zeros((np.sum(n_points),self.means[0].size))
		cumu_points = np.insert(cumu_points,0,0)
		for i in range(self.cov[:,0,0].size):
			colors[cumu_points[i]:cumu_points[i+1],:] = stats.multivariate_normal.rvs(
				mean=self.means[i,:],cov=self.cov[i,:,:].T,size=int(n_points[i]))
			
		if plot:
			plt.scatter(colors[:,0],colors[:,1],s=0.5,marker='.',color='b',alpha=0.5)
			plt.show()
			
		return colors
		
	def genMags(self,n_points=[100,100],plot=False):
		'''
		Draw the g-band magnitudes of n GCs distributed according to normal distribution
		
		n_points: array-like, optional
			i x 1 array, containing number of GCs to create for each magnitude
		'''
		cumu_points = np.cumsum(n_points)
		mags = np.zeros(np.sum(n_points))
		cumu_points = np.insert(cumu_points,0,0)
		
		#if using same luminosity for both populations generate all at once
		if self.lum_means.size < 2:
			mags = stats.norm.rvs(loc=self.lum_means,scale=self.lum_sigs,size=np.sum(n_points))
			
		#if different luminosity for both populations, generate seperaterly	
		else:
			for i in range(self.lum_means.size):
				mags[cumu_points[i]:cumu_points[i+1],:] = np.random.normal(self.lum_means[i],self.lum_sigs[i],size=int(n_points[i]))
				
		if plot:
			plt.hist(mags)
			plt.show()
			
		return mags		
				
				
	def genMock(self,n=200,fractions=np.array([1.0]),plot=False):
		'''
		Generate a 
		
		'''
		fractions = fractions / np.sum(fractions)
		n_points = np.round(n * fractions)
		
		colors = self.genColors(n_points=n_points,plot=plot)
		positions = self.genSpatial(n_points=np.sum(n_points),plot=plot)
		mags = self.genMags(n_points=n_points,plot=plot)


		return (colors,mags,positions)
		
			
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	