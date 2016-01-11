import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pickle
import colorKDE

class densityFG(object):
	'''
	A class for calculating the emperical probability density for the FG population
	for a given color. 
	
	Currently only for (g-i) and (r-i) color selection.
	'''
	
	def __init__(self,file='foreground_density.dat', smooth_stars=True,smooth_sig=False,max_rad=10.,spatial=True,red_lim=4.0,\
		blue_lim = 0.0,fg_type='kde',fg_kde_file='',faint_mag=26.0,bright_mag=18.0,center=[0.,0.],lum=False):
		
		self.fg_type = fg_type
		if self.fg_type == 'sdss_empirical':
			self.data = np.loadtxt(file)
			red_lim_ind = np.argmin(np.abs(red_lim - self.data[:,0]))
			blue_lim_ind = np.argmin(np.abs(blue_lim - self.data[:,0]))
			self.gi_color = self.data[blue_lim_ind:red_lim_ind,0]
			self.n_stars = self.data[blue_lim_ind:red_lim_ind,1]
			self.ri_color = self.data[blue_lim_ind:red_lim_ind,6]
			self.ri_sig = self.data[blue_lim_ind:red_lim_ind,7]
			if smooth_stars:
				self.smooth_stars()
			
			if smooth_sig:
				self.smooth_sig()
			
			self.star_weights = self.n_stars / np.sum(self.n_stars)
		
		elif self.fg_type == 'kde':
			self.kde = pickle.load(open(fg_kde_file,'rb'))
			
		self.max_rad = max_rad
		self.spatial = spatial
		self.faint_mag = faint_mag
		self.bright_mag = bright_mag
		self.center=center
		self.lum=lum
		
		#if the data are only complete to a fraction of the FG density,
		#need to only normalize down to this value. Should be set
		#once this value is determined in gcSampler

		self.luminosity_norm = np.log(1./(self.faint_mag-self.bright_mag))

		
		
	def read_file(self,file):
		'''
		Read in the file containing the emperical foreground densities
		'''
		return np.loadtxt(file)
		
	def smooth_stars(self,kernel_width=3):
		'''
		Smooth the n_stars distribution to remove the (real?) scatter from bin-to-bin.
		Kernel width of 3 seemed to do a good job
		'''
		self.n_stars = ndimage.filters.gaussian_filter(self.n_stars,kernel_width)
		
	def lnLikeColor(self,gi,ri):
		'''
		Returns the (un-normalized) log likelihood of the color. Treated as a gaussian with median given by ri_color
		and standard deviation given by ri_sig, then weighted by the (discreet) weights provided by star_weights
		Parameters:
			gi: array_like
				input gi color(s) of source(s)
			ri: array_like
				input ri color(s) of sources(s)
				
		Returns:
			dens: array_like
				returns individual log-likelihood of each source in an array. 
		'''	
		if self.fg_type == 'sdss_empirical':
			gi_ind = np.argmin(np.abs(np.transpose(np.tile(self.gi_color,(np.size(gi),1))) - \
				np.tile(gi,(np.size(self.gi_color),1))),axis=0)
			
			mean = self.ri_color[gi_ind]
			sig = self.ri_sig[gi_ind]
			weight = self.star_weights[gi_ind]
			dens = stats.norm.logpdf(ri,loc=mean,scale=sig) + np.log(weight) 		
			#add in normalization coefficient, should be close but needs to be corrected eventually
			return dens + np.log(50)
		
		elif self.fg_type == 'kde':
			data = np.array([gi,ri]).T
			return self.kde.score_samples(data)
		
	def LikeColor(self,gi,ri):
		'''
		conveinence function to return normal likelihood instead of log
		'''	
		gi_ind = np.argmin(np.abs(np.transpose(np.tile(self.gi_color,(np.size(gi),1))) - \
			np.tile(gi,(np.size(self.gi_color),1))),axis=0)

		mean = self.ri_color[gi_ind]
		sig = self.ri_sig[gi_ind]
		weight = self.star_weights[gi_ind]
		dens = stats.norm.pdf(ri,loc=mean,scale=sig) * weight

		print mean,sig,self.gi_color[gi_ind]

		#add in normalization coefficient, in potentially incorrect way
		return dens
		
	def lnLikeMag(self):
		'''
		Returns likelihood for a given magnitude (current assumed to be constant)
		'''
		return self.luminosity_norm
		
	def lnLikeSpatial(self):
		'''
		Returns the likelihood for a particular spatial point (currently assumed to be constant)
		'''
		return np.log(1./(2.*np.pi*self.max_rad**2))
		
	def lnLike(self,data):
		color,mags,positions = data
		gi = color[:,0]
		ri = color[:,1]
		
		mags_like = self.lnLikeMag()
		col_like = self.lnLikeColor(gi,ri)
		spatial_like = self.lnLikeSpatial()
		
		if self.lum:
			col_like = col_like + mags_like
		
		if self.spatial:
			return col_like + spatial_like
		else:
			return col_like
		
	def genColors(self,n_points=1000,plot=False):
		'''
		Generate fake gi and ri color-colors. First select gi color via inverse CDF, then select ri color via
		direct simulation.
		
		Returns:
			Tuple containing (gi, ri, x, y)
		'''
		if self.fg_type == 'sdss_empirical':
			n_stars_cumu = np.cumsum(self.n_stars) / np.sum(self.n_stars)
			gi_ind = np.argmin(np.abs(np.transpose(np.tile(n_stars_cumu,(n_points,1))) - \
				np.tile(np.random.random(n_points),(n_stars_cumu.size,1))),axis=0)
		
			#gi colors should have scatter equivlent to bin size
			gi = self.gi_color[gi_ind] + stats.uniform.rvs(size=n_points,loc=-0.01,scale=0.02)
		
			ri = stats.norm.rvs(loc=self.ri_color[gi_ind],scale=self.ri_sig[gi_ind],size=n_points)
			#ri_t = stats.t.rvs(self.n_stars[gi_ind]-1,loc=self.ri_color[gi_ind],scale=self.ri_sig[gi_ind],size=n_points)
			
		elif self.fg_type == 'kde':
			data = self.kde.sample(n_points)
			gi = data[:,0]
			ri = data[:,1]
		
		if plot:
			plt.scatter(gi,ri_t,s=0.5,marker='.',color='r',alpha=0.5)
			plt.scatter(gi,ri,s=0.5,marker='.',color='b',alpha=0.5)
			plt.show()
		
		return np.array([gi,ri]).T
	
	def genSpatial(self,n_points=1000,plot=False):
		'''
		Draw the x and y of n background sources unformly distributed in a circle of radius max_rad
		'''
		
		rho = self.max_rad * np.sqrt(np.random.uniform(0,1,n_points))
		phi = np.random.uniform(0.0,2.0*np.pi,n_points)
		x = rho*np.cos(phi)
		y = rho*np.sin(phi)
		
		x = x+self.center[0]
		y = y+self.center[1]
		
		if plot:
			plt.scatter(x,y,s=0.5,marker='.',color='b',alpha=0.5)
			plt.show()

		return np.array([x,y]).T
		
	def genMags(self,n_points,function='uniform',bright_mag=20,faint_mag=27,plot=False):
		'''
		Generate n_points magntiudes destributed according to function between bright_mag
		and faint_mag
		'''
		return stats.uniform.rvs(size=n_points) * (faint_mag-bright_mag) + bright_mag
	
	def genMock(self,n=1000):
		colors = self.genColors(n_points=n)
		positions = self.genSpatial(n_points=n)
		mags = self.genMags(n_points=n,bright_mag=self.bright_mag,faint_mag=self.faint_mag)
		
		return (colors,mags,positions)
		