import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pickle
import colorKDE
import astropy

class densityFG(object):
    '''
    A class for calculating the emperical probability density for the FG population
    for a given color.

    Currently only for (g-i) and (r-i) color selection.
    '''

    def __init__(self, smooth_stars=True,smooth_sig=False,max_rad=10.,spatial=True,red_lim=4.0,\
        blue_lim = 0.0,fg_type='kde',fg_kde_file='',faint_mag=26.0,bright_mag=18.0,center=[0.,0.],lum=False,\
        c_blue=np.array([-1.,-1.]),c_red=np.array([3.,5.]),re_norm=False,fg_mag_kde_file=None,fg_mag_complete_file=None,
        const_mag_density=True,ra_limits=[0.0,0.0],dec_limits=[0.0,0.0],radial_cut=True,const_fg_density=False):

        self.fg_type = fg_type

        self.kde = pickle.load(open(fg_kde_file,'rb'),encoding='latin1')

        self.max_rad = max_rad
        self.spatial = spatial
        self.faint_mag = faint_mag
        self.bright_mag = bright_mag
        self.center=center
        self.lum=lum
        
        self.c_blue = c_blue
        self.c_red = c_red
        
        self.ra_limits=ra_limits
        self.dec_limits=dec_limits
        self.radial_cut=radial_cut
        
        self.const_fg_density=const_fg_density

        #if the data are only complete to a fraction of the FG density,
        #need to only normalize down to this value. Should be set
        #once this value is determined in gcSampler

        self.re_norm = re_norm
        if self.re_norm:
            self.ln_color_norm = self.calcColorNorm(low_x=c_blue[0],high_x=c_red[0],low_y=c_blue[1],high_y=c_red[1])
        else:
            self.ln_color_norm = 0.0
            
        self.const_mag_density = const_mag_density
        
        #if not using luminosity, don't need to calculate normalization
        if self.lum:
            if not const_mag_density:
                self.calcFGMagNorm(fg_mag_kde_file,fg_mag_complete_file)
            else:
                self.luminosity_norm = np.log(1./(self.faint_mag-self.bright_mag))
            
        
            
            


    def read_file(self,file):
        '''
        Read in the file containing the emperical foreground densities
        '''
        return np.loadtxt(file)
        
    def calcFGMagNorm(self,fg_mag_kde_file,fg_mag_complete_file):
        """calculate the normalization for the foreground magnitude density."""
        self.fg_mag_kde = pickle.load(open(fg_mag_kde_file,'rb'),encoding='latin1')
        self.fg_mag_complete = pickle.load(open(fg_mag_complete_file,'rb'),encoding='latin1')
        grid_size = 0.01
        grid = np.arange(self.bright_mag,self.faint_mag,grid_size)
        proba = np.exp(self.fg_mag_kde.score_samples(grid.reshape(-1,1)) / self.fg_mag_complete.predict_proba(grid.reshape(-1,1))[:,1])
        self.fg_mag_norm = np.log(np.sum(proba * grid_size))
        
        
    def calcColorNorm(self,low_x=0.0,high_x=2.0,low_y=0.0,high_y=1.5,grid_size=0.02):
        """Calculate the color normalization by summing over the relevant color limits."""
        xx,yy = np.meshgrid(np.arange(low_x,high_x,grid_size),np.arange(low_y,high_y,grid_size))
        return np.log(np.sum(np.exp(self.lnLikeColor(xx.ravel(),yy.ravel()))*grid_size**2))

    def lnLikeColor(self,gi,ri):
        '''
        Returns the (un-normalized) log likelihood of the color.
        
        Parameters:
        		gi: array_like
        				input gi color(s) of source(s)
        		ri: array_like
        				input ri color(s) of sources(s)

        Returns:
        		dens: array_like
        				returns individual log-likelihood of each source in an array.
        '''
        if self.const_fg_density:
            return np.log(np.zeros(gi.shape[0]) + 1. / ((self.c_red[0] - self.c_blue[0]) * (self.c_red[1] - self.c_blue[1])))
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

        print(mean,sig,self.gi_color[gi_ind])

        #add in normalization coefficient, in potentially incorrect way
        return dens

    def lnLikeMag(self,mags):
        '''
        Returns likelihood for a given magnitude (current assumed to be constant)
        '''
        #self.mags_in_range = (mags < self.faint_mag) & (mags > self.bright_mag)
        #self.mags_in_range = self.mags_in_range.astype(int)

        if self.const_mag_density:
            return self.luminosity_norm #* self.mags_in_range
        else:
            return self.fg_mag_kde.score_samples(mags.reshape(-1,1))  -  self.fg_mag_complete.predict_log_proba(mags.reshape(-1,1))[:,1] - self.fg_mag_norm


    def lnLikeSpatial(self):
        '''
        Returns the likelihood for a particular spatial point.
        Density assumed to be constant over a circle of
        r=max_radius.
        '''
        if self.radial_cut:
            return np.log(1./(np.pi*self.max_rad**2))
        else:
            return np.log(1./((self.ra_limits[1] - self.ra_limits[0]) * (self.dec_limits[1] - self.dec_limits[0])))

    def lnLike(self,data):
        color,mags,positions = data
        gi = color[:,0]
        ri = color[:,1]

        if self.lum:
            mags_like = self.lnLikeMag(mags)
        col_like = self.lnLikeColor(gi,ri) + self.ln_color_norm
        #col_like = 0.0
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

    def genMags(self,n_points,function='uniform',bright_mag=20.,faint_mag=27.,plot=False):
        '''
        Generate n_points magntiudes destributed according to function between bright_mag
        and faint_mag
        '''
        return np.random.uniform(low=bright_mag,high=faint_mag,size=n_points)

    def genMock(self,n=1000):
        colors = self.genColors(n_points=n)
        positions = self.genSpatial(n_points=n)
        mags = self.genMags(n_points=n,bright_mag=self.bright_mag,faint_mag=self.faint_mag)

        return (colors,mags,positions)
