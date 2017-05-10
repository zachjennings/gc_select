import numpy as np
import csv
from scipy import stats
import scipy
import corner
import emcee
import pickle
import densityGC
import densityFG
import SourceCompleteness
import os
import datetime

from importlib import reload
reload(SourceCompleteness)

#Set some default priors for my own convenience, assuming 3 filters
#default_priors = np.array([[20,28],[0.2,1.5],[0.2,1.5],[0,3.0],[0,1.0],[0,1.0],[0,1],[0.1,10]])
#default_back_range = np.array([[20,28],[19,27],[18,26],[0,10],[0,2.0*np.pi]])
default_theta_init = np.array([0.5,1.0,1.0,0.0])

default_priors = np.array([[20,28],[19,27],[18,26],[0,3.0],[0,3.0],[0,3.0],[0,1],[0.01,1.0],[0,1],[0,np.pi]])
default_back_range = np.array([[20,28],[19,27],[18,26],[0,0.5],[0,np.pi]])
default_back_range_color = np.array([[0.27,1.63],[0.25,1.05],[18,26],[0,0.5],[0,2.0*np.pi]])
#default_cov_matrix = np.array([[[0.0277324,0.0173072],[0.0173072,0.0125529]]])
default_cov_matrix = np.array([[[0.04,0.015],[0.015,0.03]]])

#default_means = np.array([[0.905,0.247]])
default_means = np.array([[0.75,0.5]])

default_param_labels = ['f_gc','mu_gi','mu_ri','sig_gi','sig_ri','cov','r_s','q','pa']
default_means_bim = np.array([[0.9,0.33],[0.6,0.19]])
default_cov_matrix_bim = np.array([[[0.009, 0.0042],[0.0042, 0.0027]],\
    [[0.0082, 0.0031],[0.0031,.0018]]])
default_spatial = np.array([10.0])
default_spatial_ellipticity = np.array([10.0,0.7,0.0])
default_fractions = np.array([0.5,0.5])
default_fractions_bim = np.array([0.25,0.25,0.5])
default_lum_mean = np.array([23.])
default_lum_sig = np.array([1.4])

default_mean_priors_bimodal = np.array([[1.3,0.5],[0.5,0.1]])
default_mean_priors_single = np.array([[1.3,0.5],[0.5,0.1]])


class gcSampler(object):
    '''
    Class to run the emcee sampler for GC selection

    Attributes:
    catalog: tuple

    spatial_bimodality: bool
        toggle whether code should fit for seperate spatial distributions
        (not implemented)

    n_dim: int

    incompleteness: make data incomplete
    **fix this, currently breaks if incomplete and completeness both set**

    '''
    def __init__(self,n_walkers,catalog=np.zeros(1),n_colors=2,spatial_bimodality=False,\
        radial_profile=None,ellipticity=False,lum_function=None,n_pop=1,\
        back_range=default_back_range_color,mock=True,n_source=1000, \
        max_rad=10.0,mc_scale=2.0,fixed_cov=True,fixed_means=True,red_lim=4.0,fg_type='kde',
        fg_kde_file='cropped_color_dist.p',mock_fg_only=False,completeness='none',center=[0.,0.],\
        completeness_file='fake_gal_completeness.p',fixed_lum=False,fg_faint=25.,fg_bright=17.,\
        incomplete=False,
        fractions=np.array([]),\
        means=np.array([]),\
        spatial=np.array([]),
        cov=np.array([]),
        lum_mean=np.array([]),\
        lum_sig=np.array([]),\
        re_norm=False,c_blue=np.array([-1.,-1.]),c_red=np.array([6.,4.]),\
        ri_offset=0.0,gi_offset=0.0,gi_divide=0.85,ri_divide=0.28,
        fg_mag_kde_file='',fg_mag_complete_file='',const_fg_mag_density=False,
        ra_limits=[0.0,0.0],dec_limits=[0.0,0.0],radial_cut=True,const_fg_density=False,
        spatial_norm_grid=0.005,
        mean_priors=np.array([])):

        self.mock = mock
        self.incomplete = incomplete
        self.mock_fg_only = mock_fg_only
        self.radial_profile = radial_profile
        self.ellipticity = ellipticity
        self.n_pop = n_pop
        self.lum_function = lum_function
        self.spatial_bimodality = spatial_bimodality
        self.fg_faint = fg_faint
        self.fg_bright=fg_bright
        self.ri_offset=ri_offset
        self.gi_offset=gi_offset
        
        self.gi_divide = gi_divide
        self.ri_divide = ri_divide
        
        if mean_priors.shape[0] < 2:
            if self.n_pop == 1:
                self.mean_priors = default_mean_priors_single
            elif self.n_pop == 2:
                self.mean_priors = default_mean_priors_bimodal
        else:
            self.mean_priors = mean_priors
        

        if lum_mean.size < 1:
            lum_mean = default_lum_mean
            lum_sig = default_lum_sig

        #check if fitting for ellipticity and set approprite default parameters
        #if so
        if self.ellipticity and self.radial_profile is not None:
            if spatial.size < 1.0:
                spatial = default_spatial_ellipticity
            else:
                spatial = spatial
        elif self.radial_profile is not None:
            if spatial.size < 1.0:
                spatial = default_spatial
            else:
                spatial = spatial
        else:
            self.ellipticity = False
            spatial = np.array([1.0])

        self.fixed_lum = fixed_lum
        if self.lum_function is not None:
            self.lum = True
        else:
            self.lum = False

        #check if bimodality will be fixed
        self.fixed_cov = fixed_cov
        self.fixed_means = fixed_means
        if self.n_pop==2:
            
            if cov.size < 1:
                cov = default_cov_matrix_bim
            if means.size < 1:
                means = default_means_bim
                
            means[:,0] = means[:,0] + self.gi_offset
            means[:,1] = means[:,1] + self.ri_offset
            self.fractions = default_fractions_bim
            self.bimodality = True
        else:
            if cov.size < 1:
                cov = default_cov_matrix_bim
            if means.size < 1:
                means = default_means_bim
            #cov = default_cov_matrix
            #means = default_means
            self.fractions = default_fractions
            self.bimodality = False

        if fractions.size > 0:
            self.fractions = fractions

        self.means=means
        self.cov = cov
        self.completeness=completeness
        self.n_source = n_source
        self.n_walkers = int(n_walkers)

        self.fractions = fractions/np.sum(fractions)
        self.n_fg = int(np.round(fractions[-1] * self.n_source))
        self.n_gc = int(self.n_source - self.n_fg)

        #n_pop = number of populations of GCs to look for
        self.n_pop = n_pop

        #n_colors = number of colors available
        self.n_colors = n_colors

        #initialize the density objects, as desired
        if self.radial_profile is None or self.radial_profile=='none':
            self.fg = densityFG.densityFG(max_rad=max_rad,spatial=False,fg_type=fg_type,fg_kde_file=fg_kde_file,center=center,lum=self.lum,\
                                            faint_mag=self.fg_faint,bright_mag=self.fg_bright,re_norm=re_norm,c_blue=c_blue,c_red=c_red,
                                            fg_mag_kde_file=fg_mag_kde_file,fg_mag_complete_file=fg_mag_complete_file,const_mag_density=const_fg_mag_density,
                                            ra_limits=ra_limits,dec_limits=dec_limits,radial_cut=radial_cut,const_fg_density=const_fg_density)

        else:            
            self.fg = densityFG.densityFG(max_rad=max_rad,red_lim=red_lim,fg_type=fg_type,fg_kde_file=fg_kde_file,center=center,lum=self.lum,\
                                            faint_mag=self.fg_faint,bright_mag=self.fg_bright,re_norm=re_norm,c_blue=c_blue,c_red=c_red,\
                                            fg_mag_kde_file=fg_mag_kde_file,fg_mag_complete_file=fg_mag_complete_file,const_mag_density=const_fg_mag_density,
                                            ra_limits=ra_limits,dec_limits=dec_limits,radial_cut=radial_cut,const_fg_density=const_fg_density)

        self.gc = densityGC.densityGC(radial_profile=self.radial_profile,ellipticity=self.ellipticity,\
            fixed_cov=self.fixed_cov,means=means,cov=cov,spatial=spatial,center=center,lum=self.lum,fixed_lum=self.fixed_lum,
            lum_mean=lum_mean,lum_sig=lum_sig,fg_faint=self.fg_faint,fg_bright=self.fg_bright,re_norm=re_norm,c_blue=c_blue,c_red=c_red,\
            fixed_means=fixed_means,max_rad=max_rad,const_mag_density=True,ra_limits=ra_limits,dec_limits=dec_limits,radial_cut=radial_cut,
            spatial_norm_grid=spatial_norm_grid)

        #generate mock data if requested, otherwise use provided catalog
        if self.mock or self.mock_fg_only:
            self.MakeMock()

        else:
            self.data = catalog

        #check if including completeness corrections, and create completeness object if needed
        if self.completeness == 'full':
            completeness_obj_g = pickle.load(open(completeness_file,'rb'))
            completeness_obj_r = pickle.load(open(completeness_file,'rb'))
            completeness_obj_i = pickle.load(open(completeness_file,'rb'))

            self.norm = SourceCompleteness.CompleteNormalization(self.gc,self.fg,\
                           [completeness_obj_g,completeness_obj_r,completeness_obj_i],filters=3,fixed_cov=self.fixed_cov,fixed_lum=self.fixed_lum,\
                            fg_faint=self.fg_faint,fg_bright=self.fg_bright,n_pop=self.n_pop,c_blue=c_blue,c_red=c_red)

            self.fg_complete_norm = self.norm.calc_fg_norm()

            if self.fixed_cov and self.fixed_lum:
                #since fractions will matter for normalization, need to set fractions to full and calibrate later
                self.gc_complete_norm = self.norm.calc_gc_norm(None,None,None,None,self.fractions[:-1],first_run=True)

            elif self.fixed_cov:
                self.gc_complete_norm = self.norm.calc_gc_norm(None,None,lum_mean,lum_sig,self.fractions[:-1],first_run=True)


            i = self.data[1]
            g = self.data[0][:,0] + i
            r = self.data[0][:,1] + i

            g_comp = completeness_obj_g.QueryComplete(g)[:,0]
            r_comp = completeness_obj_r.QueryComplete(r)[:,0]
            i_comp = completeness_obj_i.QueryComplete(i)[:,0]

            self.ln_complete = g_comp + r_comp + i_comp
            
        elif self.completeness == 'mag_only':
            completeness_obj = pickle.load(open(completeness_file,'rb'),encoding='latin1')

            self.norm = SourceCompleteness.CompleteNormalization(self.gc,self.fg,\
                           [completeness_obj],filters=1,fixed_cov=self.fixed_cov,fixed_lum=self.fixed_lum,\
                            fg_faint=self.fg_faint,fg_bright=self.fg_bright,n_pop=self.n_pop,c_blue=c_blue,c_red=c_red)

            self.fg_complete_norm = self.norm.calc_fg_norm()

            if self.fixed_cov and self.fixed_lum:
                self.gc_complete_norm = self.norm.calc_gc_norm(None,None,None,None,self.fractions[:-1])

            self.ln_complete = completeness_obj.predict_log_proba(self.data[1].reshape(-1,1))[:,1]
            '''
            i = self.data[1]
            g = self.data[0][:,0] + i
            r = self.data[0][:,1] + i
            
            completeness_obj_g = pickle.load(open(completeness_file,'rb'))
            completeness_obj_r = pickle.load(open(completeness_file,'rb'))
            completeness_obj_i = pickle.load(open(completeness_file,'rb'))

            g_comp = completeness_obj_g.QueryComplete(g)[:,0]
            r_comp = completeness_obj_r.QueryComplete(r)[:,0]
            i_comp = completeness_obj_i.QueryComplete(i)[:,0]

            self.ln_complete = g_comp + r_comp + i_comp
            '''

        elif self.completeness != 'full' and self.completeness != 'mag_only':
            self.fg_complete_norm = 1.
            self.gc_complete_norm = 1.
            self.ln_complete = 0.

        #initialize the sampler
        self.n_dim = self.n_pop + int(self.radial_profile is not None) + \
            2*int(self.ellipticity) + (3*int(not self.fixed_cov) + 2*int(not self.fixed_means))*(self.n_pop) +\
            2*int(not self.fixed_lum)
        self.sampler = emcee.EnsembleSampler(self.n_walkers,self.n_dim,\
            self.lnLike,a=mc_scale)

        #if we're dealing with mock data, make the data appropriately incomplete
        if self.mock and (self.completeness is not None or self.incomplete):
            self.MockIncomplete(completeness_file)

        #Since we're assuming the FG distribution is fixed, we can just calculate the
        #log likelihood once and use that for the rest of the samples
        self.fg_lnlike = self.fg.lnLike(self.data)

        #if we're fitting to mock data, need to fix fractions to account for the number of GCs we now have
        #unless we're modelling incompleteness.
        if self.mock:
            new_n_gc = np.sum(self.gc_array)
            new_n_fg = np.sum(np.abs(self.gc_array - 1))
            gc_complete_frac = new_n_gc / self.n_gc
            fg_complete_frac = new_n_fg / self.n_fg
            self.new_n_gc = new_n_gc
            self.new_n_fg = new_n_fg
            #fractions[:-1] = fractions[:-1] * gc_complete_frac
            #fractions[-1] = fractions[-1] * fg_complete_frac
            #fractions = fractions/np.sum(fractions)

        #need to shape fractions,means,covariances, etc into correct shape for theta_init
        self.theta_init = self.thetaRepack(fractions,means,cov,spatial,lum_mean,lum_sig)
        self.pos = np.array([(self.theta_init + 1.0e-4*np.random.randn(self.n_dim) + self.theta_init*1.0e-4*np.random.randn(self.n_dim)) \
                            for j in range(self.n_walkers)])

    def MakeMock(self):
        """
        Function to make mock catalogs from FG and GC objects.
        """
        if self.mock:
            mock_gc_color,mock_gc_mags,mock_gc_coordinates = self.gc.genMock(n=self.n_gc,fractions=self.fractions[:-1])
            mock_fg_color,mock_fg_mags,mock_fg_coordinates = self.fg.genMock(n=self.n_fg)

            #create tracker array to keep track of how many GCs and contaminants are added
            self.gc_array = np.concatenate([np.ones(self.n_gc),np.zeros(self.n_fg)])

            colors = np.concatenate((mock_gc_color,mock_fg_color))
            mags = np.concatenate((mock_gc_mags,mock_fg_mags))
            coordinates = np.concatenate((mock_gc_coordinates,mock_fg_coordinates))
            self.data = (colors,mags,coordinates)

        elif self.mock_fg_only:
            #test given color data from N3115. Currently supplying real coordinates
            #for GCs with mock profile
            mock_fg_color,mock_fg_mags,mock_fg_coordinates = self.fg.genMock(n=self.n_fg)
            real_color,real_mags,real_coordinates = catalog
            #real_coordinates = self.gc.genSpatial(n_points=real_color.shape[0])
            colors = np.concatenate((real_color,mock_fg_color))
            mags = np.concatenate((real_mags,mock_fg_mags))
            coordinates = np.concatenate((real_coordinates,mock_fg_coordinates))
            self.data = (colors,mags,coordinates)

    def MockIncomplete(self, completeness_file):
        '''
        Make mock data incomplete if we are dealing with incompleteness.

        Current strategy is just to randomly select from bernoulli RVs.
        '''
        #if we're setting imcomplete data for testing purposes,
        #need to calculate appropriate incompleteness here
        if self.incomplete:
            completeness_obj = pickle.load(open(completeness_file, 'rb'))
            ln_complete = completeness_obj.QueryComplete(self.data[1])[:, 0]
            comp = stats.bernoulli.rvs(p=np.exp(ln_complete))

        #otherwise, use current completeness array and then
        #discared unselected GCs
        else:
            comp = stats.bernoulli.rvs(p=np.exp(self.ln_complete))
            self.ln_complete = self.ln_complete[comp > 0.5]

        comp_color = self.data[0][comp > 0.5,:]
        comp_mag = self.data[1][comp > 0.5]
        comp_spatial = self.data[2][comp > 0.5,:]
        self.data = (comp_color,comp_mag,comp_spatial)
        self.gc_array = self.gc_array[comp > 0.5]


    def thetaRepack(self,fractions,means,cov,spatial,lum_mean,lum_sig):#,total_sources):
        '''
        Repacks the various arrays of parameters into theta for MCMC

        Inputs
        ---------
        theta_un: tuple
            contains (fractions,means,covariances,spatial)

        Returns
        -------
        theta_unpacked: 1d array
            array containing all parameters in correct order for MCMC

        '''
        theta = fractions[:-1]
        for j in range(self.n_pop):
            if not self.fixed_means:
                theta = np.concatenate([theta,means[j]])
        
            if not self.fixed_cov:
                cov_array = np.array([cov[j,0,0],cov[j,1,1],cov[j,1,0]])
                theta = np.concatenate([theta,cov_array])

        if self.radial_profile is not None:
            theta = np.concatenate([theta,spatial])

        if self.lum_function is not None and not self.fixed_lum:
            theta = np.concatenate([theta,lum_mean,lum_sig])

        #if self.completness is not None:
        #   theta = np.concatenate([total_sources])

        return theta

    def thetaUnpack(self,theta):
        '''
        Unpacks the 1d array theta into more tractable groups

        Inputs
        ---------
        theta: 1d array
            contains current values for all dimensions

        Returns
        -------
        theta_unpacked: tuple
            of form (fractions,means,covariance matricies,scale parameters)

        '''
        #value to track theta array index
        i = 0

        #first n-1 values in theta are fractions for n components
        fractions = theta[:self.n_pop]
        fractions = np.append(fractions,[1.-np.sum(fractions)])
        i += self.n_pop

        #next 2 x n_bands + n_bands + 1 values will correspond to covariance matrix *if* fixed_cov is not set
        cov = np.zeros((self.n_pop,self.n_colors,self.n_colors))
        means = np.zeros((self.n_pop,self.n_colors))
        for j in range(self.n_pop):
            if not self.fixed_means:
                means[j] = np.array([theta[i],theta[i+1]])
                i += self.n_colors
                
                if not self.fixed_cov:
                    cov[j] = np.array([[theta[i],theta[i+2]],[theta[i+2],theta[i+1]]])
                    i+= self.n_colors + 1
                
                    
            elif not self.fixed_cov:
                cov[j] = np.array([[theta[i],theta[i+2]],[theta[i+2],theta[i+1]]])
                i+= self.n_colors + 1

        #next, unpack radial profiles, checking if user wants to consider
        #bimodality in spatial populations too
        if self.radial_profile is not None and self.spatial_bimodality:
            spatial = np.ones((self.n_pop,3))

            for j in range(self.n_pop):
                #if elliptical, then each profile has 3 parameters, otherwise
                #each profile only has one
                if self.ellipticity:
                    spatial[j,:] = theta[i:i+3]
                    i+=3
                else:
                    spatial[j] = theta[i]
                    i+=1

        #if not bimodal, then can fill spatial easily
        #still need to check ellipticity
        elif self.radial_profile is not None:
            if self.ellipticity:
                spatial = theta[i:i+3]
                i+=3
            else:
                spatial = np.array([theta[i]])
                i+=1

        else:
            spatial = np.ones(3)
            spatial[2] = 0.

        #magnitudes will all go at the end of theta
        if self.lum_function is not None and not self.fixed_lum:
            lum_mean = theta[i:i+1]
            lum_sig = theta[i+1:i+2]
            i+=2
        else:
            lum_mean = np.ones(1)
            lum_sig = np.ones(1)
            i+=2

        #if self.completeness is not None:
        #   total_sources = theta[i]

        return (fractions,means,cov,spatial,lum_mean,lum_sig)#,total_sources)


    def lnLike(self,theta):
        '''
        Calculates log likelihood, given current parameter values theta
        and using data stored in self.data

        Inputs
        ---------
        theta: 1d array
            contains current values for all dimensions

        Returns
        -------
        lnlike: float
            current log likelihood + log prior

        '''
        fractions,means,cov,spatial,lum_mean,lum_sig = self.thetaUnpack(theta)

        #calculate priors of population fractions
        lp_frac = self.lnPriorFractions(fractions)

        #check if we're keeping covariance fixed and calculate priors
        if not self.fixed_cov:
            lp_cov = self.lnPriorCov(cov)
        else:
            lp_cov = 0.0
            
        if not self.fixed_means:
            lp_means = self.lnPriorMeans(means)
        else:
            lp_means = 0.0

        #check if we're modelling spatial distributions and calculate priors
        if self.radial_profile is not None:
            lp_spatial = self.lnPriorSpatial(spatial)
        else:
            lp_spatial = 0.0

        #check if modelling luminosity function
        if self.lum and not self.fixed_lum:
            lp_lum_function = self.lnPriorLumMean(lum_mean) + self.lnPriorLumSig(lum_sig)
        else:
            lp_lum_function = 0.0
            

        #calculate likelihood for sampling uncertainty of
        #currently modelled as Poisson, may be "wrong but good enough"
        #if self.completeness is not None:
        #   ln_like_n_pred = stats.poisson.logpdf(self.data[1].size,total_sources)
        #   lp_n_pred = 0.0  #still need to implement completeness function
        #else:
        #   ln_like_n_pred = 0.0
        #   lp_n_pred = 0.0

        lp = lp_frac + lp_cov + lp_spatial + lp_lum_function + lp_means

        if not np.isfinite(lp):
            return -np.inf

        self.lnlike_gc =  self.gc.lnLike(self.data,fractions=fractions[:-1],spatial=spatial,cov=cov,means=means,
                                   lum_means=lum_mean,lum_sigs=lum_sig)

        #calculate completeness normalization for GC data, which is dependent on theta
        if self.completeness is not None:
            # if the distributions are being kept fixed, just need to scale the already-calculated normalization
            if self.fixed_lum and self.fixed_cov:
                self.current_gc_complete_norm = self.norm.calc_gc_norm(None,None,None,None,fractions[:-1])

            #if distributions are being fit for, need to calculate normalization based on the new parameters
            else:
                self.current_gc_complete_norm = self.norm.calc_gc_norm(means,cov,lum_mean,lum_sig,fractions[:-1])

            #obsolete now that we are always passing fractions to completeness object
            #if self.completeness == 'mag_only':
            #   self.current_gc_complete_norm = np.sum(fractions[:-1]) * self.current_gc_complete_norm

            self.completeness_norm = fractions[-1]*self.fg_complete_norm + self.current_gc_complete_norm
            self.completeness_norm_ln = np.log(self.completeness_norm)

        else:
            self.completeness_norm_ln = 0.0

        lnlike_fg = np.log(fractions[-1]) + self.fg_lnlike

        lnlike = np.sum(np.logaddexp(lnlike_fg,self.lnlike_gc) + self.ln_complete - self.completeness_norm_ln)

        return lnlike + lp


    def lnPriorLumMean(self,mean):
        '''
        Calculate priors for luminosity mean

        Currently only works for single mean
        '''
        if mean < 18 or mean > 25:
            return -np.inf
        return 0.0

    def lnPriorLumSig(self,sigma):
        '''
        Calculate priors for luminosity sigma

        Currently only works for single population
        '''
        if sigma < 0.0000001:
            return -np.inf
        else:
            return -2.*np.log(sigma)


    def lnPriorCov(self,cov):
        '''
        Calculate priors for covariance parameters.
        Currently only works for 2-band, uni or bimodal data.

        '''
        cov_0 = cov[0]

        #test range of covariance terms in matrix 0
        if  cov_0[0,0] < 0.000001 or cov_0[0,0] > 0.3 or  cov_0[1,1] < 0.000000 or cov_0[1,1] > 0.3 \
             or cov_0[1,0] < 0.000001 or np.abs(cov_0[1,0]**2 / cov_0[0,0] / cov_0[1,1]) > 1.0:
            lp_cov_0 = -np.inf
        else:
            #lp_cov_0 = np.log(np.linalg.det(cov_0)**(-1.5))
            lp_cov_0 = stats.wishart.logpdf(cov_0,3.,self.cov[0,:,:])
            #np.log(1./(cov_0[0,0] * cov_0[1,1] * (1.-cov_0[1,0])**2))
            #lp_cov_0 = 0.

        if self.bimodality:
            cov_1 = cov[1]
            #test range of covariance terms in matrix 1
            if  (cov_1[0,0] < 0.000001) or (cov_1[0,0] > 0.2) or cov_1[1,1] < 0.000000 or cov_1[1,1] > 0.2  \
             or cov_1[1,0] < 0.000001 or np.abs(cov_1[1,0]**2 / cov_1[0,0] / cov_1[1,1]) > 1.0:
                lp_cov_1 = -np.inf
            else:
                #lp_cov_1 = np.log(np.linalg.det(cov_1)**(-1.5))
                lp_cov_1 = stats.wishart.logpdf(cov_1,3.,self.cov[1,:,:])
                #np.log(1./(cov_1[0,0] * cov_1[1,1] * (1.-cov_1[1,0])**2))
                #lp_cov_1=0.

        else:
            lp_cov_1 = 0.0

        return lp_cov_0 + lp_cov_1

    def lnPriorMeans(self,means):
        '''
        Calculate priors for covariance parameters.
        Currently only works for 2-band bimodal and unimodal data.

        '''
        means_0 = means[0]

        if self.n_pop==1:
        #test location of mean terms in means_0
            if  (means_0[0] > self.mean_priors[0,0]) or (means_0[0] < self.mean_priors[1,0]) or (self.mean_priors[1,1] > means_0[1]) or (means_0[1] > self.mean_priors[1,0]):
                lp_means_0 = -np.inf
            else:
                lp_means_0 = 0.0
            return lp_means_0

        else:
            means_1 = means[1]
            
            if (means_0[0] < self.gi_divide) or (self.gi_divide < means_1[0]) or (means_1[0] < self.mean_priors[1,0]) or (means_1[1] < self.mean_priors[1,1]) or \
                (means_0[0] > self.mean_priors[0,0]) or (means_0[1] > self.mean_priors[0,1]) or (means_0[1] < means_1[1]) or (means_0[1] < self.ri_divide) or (self.ri_divide < means_1[1]):
                lp_means_0 = -np.inf
            else:
                lp_means_0 = 0.0
            
            #if (means_0[0] < 0.85 + self.gi_offset) or (means_0[1] < 0.23 + self.ri_offset) or (1.5 + self.gi_offset < means_0[0]) or (0.7 + self.ri_offset < means_0[1]):
            #    lp_means_0 = -np.inf
                
            #if (means_0[0] < self.gi_divide + self.gi_offset) or (1.5 + self.gi_offset < means_0[0]):
            #    lp_means_0 = -np.inf
            #else:
                #lp_means_0 = stats.norm.logpdf(means_0[0],loc=self.theta_init[2],scale=1000) + stats.norm.logpdf(means_0[1],loc=self.theta_init[3],scale=1000)
            #    lp_means_0 = 0.0    
                    
        #test location of mean terms in means_1
            #if  (means_1[0] > self.gi_divide + self.gi_offset) or (0.2 + self.gi_offset > means_1[0]):
            #    lp_means_1 = -np.inf
                
            #if  (means_1[0] > 0.85 + self.gi_offset) or (means_1[1] > 0.23 + self.ri_offset) or (0.5 + self.gi_offset > means_1[0]) or (-0.05 + self.ri_offset > means_1[1]):
            #    lp_means_1 = -np.inf
            #else:
                #lp_means_1 = stats.norm.logpdf(means_1[0],loc=self.theta_init[7],scale=1000) + stats.norm.logpdf(means_1[1],loc=self.theta_init[8],scale=1000)
            lp_means_1 = 0.0

        return lp_means_0 + lp_means_1

    def lnPriorFractions(self,fractions):
        if np.any(fractions < 0.0) or np.any(fractions > 1.0) or (np.sum(fractions) > 1.0):
            return -np.inf
        return stats.dirichlet.logpdf(fractions,np.zeros(len(fractions)) +1./len(fractions))
        #return 0.0

    def lnPriorLuminosity(self,mean):
        if mean[0] < 18 or lum[0] > 25.0:
            return -np.inf
        return -np.log(lum[1])

    def lnPriorSpatial(self,spatial):

        #check scale radius prior, returning non-informative if range OK
        if (spatial[0] > 0.0) & (spatial[0] < 1.0):
            lp_r_s = np.log(1./spatial[0])
        else:
            lp_r_s = -np.inf

        #check range of ellipticity:
        if self.ellipticity:
            if 1.0 >= spatial[1] > 0.0:
                lp_q = 0.0
            else:
                lp_q = -np.inf

            if -np.pi/4. < spatial[2] < 3.*np.pi/4.:
                lp_pa = 0.0
            else:
                lp_pa = -np.inf
        else:
            return lp_r_s

        return lp_r_s + lp_q + lp_pa

    def lnPrior(self,theta):
        #verify f_gc is in correct range
        f_gc = theta[0]
        if  0.0 < f_gc < 1.0:
            f_gc_prior = 0.0
        else: f_gc_prior = -np.inf

        #if radial profile is being fit, incorporate uninformative prior for scale radius
        if self.radial_profile != 'none':
            r_s = theta[1]
            if 100. < r_s < 1e8:
                r_s_prior = np.log(1./r_s**2)
            else:
                r_s_prior = -np.inf
        else: r_s_prior = 0.0

        #if ellipticity is being fit, verify ellipticity and pa are in reasonable range
        if self.ellipticity == True:
            q = theta[2]
            pa = theta[3]
            if 0.0 < q < 1.0:
                q_prior = 0.0
            else:
                q_prior = -np.inf

            if 0.0 < pa < np.pi:
                pa_prior = 0.0
            else:
                pa_prior = -np.inf
        else:
            q_prior = 0.0
            pa_prior = 0.0

        return f_gc_prior + r_s_prior + q_prior + pa_prior

    def runSampler(self,n_steps,plot=False,burn_steps=0.,print_stats=True):
        self.sampler.run_mcmc(self.pos,n_steps)
        self.pos = self.sampler.chain[:,-1,:]


        self.chain=self.sampler.chain[:,burn_steps:,:]

        if plot:
            self.makeTriangle(n_steps/5)

        if print_stats:
            self.calcStats()

    def makeTriangle(self,burn_in,fig_name='foo.png'):
        chain = self.sampler.chain[:,burn_in:,:].reshape(-1,self.n_dim)
        if self.mock:
            fig = triangle.corner(chain)
        else:
            fig = triangle.corner(chain)
        fig.savefig(fig_name)
        fig.clear()

    def calcStats(self,print_stats=True,print_pgcs=False,high_range=84,low_range=16,n_burn=0.):
        '''
        Calculates medians and Confidence intervals the parameters from self.chain
        '''
        flat_chain = self.chain[:,n_burn:,:].reshape(-1,self.n_dim)

        self.medians = np.median(flat_chain,axis=0)
        self.lows = np.percentile(flat_chain,low_range,axis=0)
        self.highs = np.percentile(flat_chain,high_range,axis=0)

        self.tot_gcs = np.sum(self.medians[self.n_dim:])
        self.tot_fg = np.sum(1.-self.medians[self.n_dim:])

        fractions,means,cov,spatial,lum_mean,lum_sig = self.thetaUnpack(self.medians)

        lnLikeGC = self.gc.lnLike(self.data,fractions=fractions[:-1],spatial=spatial,cov=cov,means=means,lum_means=lum_mean,lum_sigs=lum_sig)
        lnLikeFG = np.log(fractions[-1]) + self.fg.lnLike(self.data)
        self.p_gc = np.exp(lnLikeGC - np.logaddexp(lnLikeGC,lnLikeFG))
        self.p_fg = 1.-self.p_gc

        if self.mock:
            self.false_neg = self.p_gc[:self.n_gc-1] < 0.5
            self.false_pos = self.p_gc[self.n_gc:] > 0.5

        #Calculate median probabilities for each GC
        #if self.radial_profile == 'none':
        #   lnLikeGC = np.log(self.medians[0]) +  self.gc.lnLike(self.data)
        #else:
        #   lnLikeGC = np.log(self.medians[0]) +  self.gc.lnLike(self.data,r_s = self.medians[1])

        #lnLikeFG = np.log(1. - self.medians[0]) + self.fg.lnLike(self.data)

        #self.p_gc = np.exp(lnLikeGC - np.logaddexp(lnLikeGC,lnLikeFG))
        #self.p_fg = 1.-self.p_gc

        #if self.mock:
        #   self.false_neg = self.p_gc[:self.n_gc-1] < 0.5
        #   self.false_pos = self.p_gc[self.n_gc:] > 0.5

        #Print out quantities

        if print_pgcs:
            for i in np.arange(self.n_obs):
                j = i+self.n_dim-self.n_obs
                print('p_gc',str(i),' median, 2.5%, and 97.5%:', self.medians[j],'  ',\
                    self.lows[j],'  ',self.highs[j])

        return (self.medians,self.lows,self.highs)
        
    def saveChain(self,filename=None,mkdir=True):
        '''
        Save the current chain in a .npy format. 
        '''
        
        np.save(filename,self.sampler.chain)

    def getParamLabels(self):
        if self.n_pop == 1 and self.fixed_cov and not self.lum:
            return ['f_gc','r_s','q','pa']
        elif self.n_pop == 1 and not self.fixed_cov and not self.lum:
            return ['f_gc','mu_gi','mu_ri','sig_gi','sig_ri','cov','r_s','q','pa']
        elif self.n_pop == 2 and self.fixed_cov and not self.lum:
            return ['f_gc_1','f_gc_2','r_s','q','pa']
        elif self.n_pop == 2 and not self.fixed_cov and not self.lum:
             return ['f_gc','f_gc_2','mu_gi_1','mu_ri_1','sig_gi_1','sig_ri_1','cov_1',\
            'mu_gi_2','mu_ri_2','sig_gi_2','sig_ri_2','cov_2','r_s','q','pa']
        elif self.n_pop == 2 and not self.fixed_cov and self.lum:
             return ['f_gc','f_gc_2','mu_gi_1','mu_ri_1','sig_gi_1','sig_ri_1','cov_1',\
            'mu_gi_2','mu_ri_2','sig_gi_2','sig_ri_2','cov_2','r_s','q','pa','mu_g','sig_g']

        else:
            return np.chararray(100) + 'foo'

class gcRunner(object):
    def __init__(self,n_walkers,f_gc,n_sources,r_s,q,pa):
        self.f_gc = f_gc
        self.n_sources = n_sources
        self.r_s = r_s
        self.q = q
        self.pa = np.float(pa) * np.pi
        self.n_dim = 4
        self.n_walkers = n_walkers
        self.filename = 'logs/nsource='+str(n_sources) + '_fgc=' + str(f_gc) + '_rs=' + \
            str(r_s) + '_q=' + str(q) + '_pa=' + str(pa) +'pi.log'

    def initSampler(self):
        n_gc = np.round(self.n_sources * self.f_gc)
        n_fg = np.round(self.n_sources * (1.-self.f_gc))
        return gcSampler(self.n_walkers,n_gc=n_gc,n_fg=n_fg,r_s=self.r_s,q=self.q,pa=self.pa,mock=True,radial_profile='exponential',ellipticity=True,\
            theta_init=[self.f_gc,self.r_s,self.q,self.pa])

    def runSampler(self,n_runs,n_steps=5000,n_burn=2500):
        self.run_stats = np.zeros((n_runs,self.n_dim*3))

        for i in range(n_runs):
            self.this_sampler = self.initSampler()

            self.this_sampler.runSampler(n_steps,burn_steps=n_burn,print_stats=False)
            self.stats = self.this_sampler.calcStats(print_stats=False)

            self.run_stats[i,:self.n_dim] = self.stats[0]
            self.run_stats[i,self.n_dim:2*self.n_dim] = self.stats[1]
            self.run_stats[i,2*self.n_dim:3*self.n_dim] = self.stats[2]

            if (100./n_runs)*i%5 == 0 and i != 0:
                print(str(100/n_runs*i) + '% Complete')

        with open(self.filename,"w+") as the_file:
            for i in self.run_stats:
                the_file.write(np.array2string(i,max_line_width=np.inf) + '\n')


class WriteSampler(object):
    """Class to write out stats and the chain for the sampler."""
    def __init__(self,sampler,prefix='/Users/zacharyjennings/astro/gc_select_logs/'):
        self.sampler = sampler
        self.prefix = prefix

        self.check_parameters()
        self.check_path()


            
    def write(self,chain=True,catalog=True,stats=True):
        """get the time and write files of interes"""
        time =  str(datetime.datetime.now()).split()
        time = time[0]+'_'+time[1]

        if chain:
            self.save_chain(time=time)

        if catalog:
            self.save_catalog(time=time)

        if stats:
            self.write_stats(time=time)

    def check_parameters(self):
        '''
        Check the relevant parameters of the sampler and input them in the paths
        '''
        self.completeness = self.sampler.completeness
        if self.completeness is None:
            self.completeness='no_completeness_'
        elif self.completeness == 'mag_only':
            self.completeness = ''


        self.spatial = self.sampler.radial_profile
        if self.spatial is None:
            self.spatial = 'no_spatial_'
        else:
            self.spatial='free_spatial_'

        if self.sampler.fixed_cov:
            self.fixed_cov = 'fixed_cov_'
        else:
            self.fixed_cov='free_cov_'

        if self.sampler.lum_function == 'mag_only':
            self.lum = 'single_lum'
        else:
            self.lum = 'no_lum'

        self.folder_path = self.prefix + self.completeness + self.spatial + self.fixed_cov + self.lum +'/'

    def check_path(self):
        '''
        Check to see if the folder path exists, make if it doesn't.
        '''
        if not os.path.isdir(self.folder_path):
            os.makedirs(self.folder_path)
            
    def save_chain(self,time=''):
        """save the MCMC chain from the sampler in a .npy format"""
        np.save(self.folder_path+'chain_'+time+'.npy',self.sampler.sampler.chain)
        
    def write_stats(self,time=''):
        """write out descriptive statistics of the chain"""
        pass
        
    def save_catalog(self,time=''):
        """save the catalog from the sampler in a .npy format"""
        color = self.sampler.data[0]
        mags = self.sampler.data[1]
        spatial = self.sampler.data[2]
        np.save(self.folder_path+'color_'+time+'.npy',color)
        np.save(self.folder_path+'mags_'+time+'.npy',mags)
        np.save(self.folder_path+'spatial_'+time+'.npy',spatial)
        

def fileRead(filename):
    '''
    Convenience function to read in an ascii catalog
    '''
    f = file(filename, 'r')
    gr=list()
    gi=list()
    ra = list()
    dec = list()
    for line in f:
        line = line.strip()
        columns = line.split()
        gr.append(float(columns[0]))
        gi.append(float(columns[1]))
        #d.append(float(columns[2]))
        ra.append(float(columns[2]))
        dec.append(float(columns[3]))
    gr = np.array(gr)
    gi = np.array(gi)
    #d = np.array(d)
    ra = np.array(ra)
    dec = np.array(dec)
    return (gr,gi,ra,dec)

def readTestData():
    data = np.loadtxt('n3115_spec_conf.txt')
    spatial = np.zeros(data.shape)
    return (data,spatial)