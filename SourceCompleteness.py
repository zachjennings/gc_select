from sklearn.svm import SVC as svc
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import pickle
import numpy as np


class SourceCompleteness(object):
        '''
        Class to hold the completeness function for the dataset in question

        '''

        def __init__(self,fake_star_catalog,spatial=False,scale='standard'):
                self.scale=scale
                self.spatial = spatial
                self.det,self.data = self.Classify(fake_star_catalog)


        def FitScale(self,data):
                self.mag = data[0,:]
                self.x = data[1,:]
                self.y = data[2,:]

                if self.scale == 'standard':
                        self.x_scaler = StandardScaler()
                        self.y_scaler = StandardScaler()
                        self.mag_scaler = StandardScaler()

                        scale_x = self.x_scaler.fit_transform(self.x)
                        scale_y = self.y_scaler.fit_transform(self.y)
                        scale_mag = self.mag_scaler.fit_transform(self.mag)

                elif self.scale == 'range':
                        scale_x = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
                        scale_y = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
                        scale_mag = (self.mag - np.min(self.mag)) / (np.max(self.mag) - np.min(self.mag))

                return (scale_mag,scale_x,scale_y)

        def Scale(self,data):
                if self.scale == 'standard':
                        scale_mag = self.mag_scaler.transform(data[0,:])
                        scale_x = self.x_scaler.transform(data[1,:])
                        scale_y = self.y_scaler.transform(data[2,:])

                elif self.scale == 'range':
                        scale_mag = (data[0,:] - np.min(self.mag)) / (np.max(self.mag) - np.min(self.mag))
                        scale_x = (data[1,:] - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
                        scale_y = (data[2,:] - np.min(self.y)) / (np.max(self.y) - np.min(self.y))

                return (scale_mag,scale_x,scale_y)

        def makeSVM(self,**kwargs):
                scale_mag,scale_x,scale_y = self.FitScale(self.data)

                self.scaled_data = np.c_[scale_mag,scale_x,scale_y]

                self.completeness = svc(probability=True,**kwargs)

                if self.spatial:
                        self.completeness.fit(self.scaled_data,self.det.ravel())
                else:
                        self.completeness.fit(np.c_[scale_mag],self.det.ravel())

        def QueryComplete(self,mag=np.array([]),ra=np.array([]),dec=np.array([])):
                if ra.size < 1 and dec.size < 1 and not self.spatial:
                        ra = np.ones(mag.shape)
                        dec = np.ones(mag.shape)
                data = np.c_[mag,ra,dec]
                scale_mag,scale_x,scale_y = self.Scale(data.T)
                if self.spatial:
                        return self.completeness.predict_log_proba(np.c_[scale_mag,scale_x,scale_y])
                else:
                        return self.completeness.predict_log_proba(np.c_[scale_mag])

        def Classify(self,fake_stars,tol=2.):
                diff = np.array([])
                mag = np.array([])
                y = np.array([])
                x = np.array([])
                for i in fake_stars.fake_cats.keys():
                        diff = np.append(diff,fake_stars.fake_cats[i][0,:] - fake_stars.fake_cats[i][1,:])
                        mag = np.append(mag,fake_stars.fake_cats[i][0,:])
                        x = np.append(x,fake_stars.fake_cats_x[i][:])
                        y = np.append(y,fake_stars.fake_cats_y[i][:])

                mag = np.array([mag]).T
                y = np.array([y]).T
                x = np.array([x]).T
                data = np.hstack((mag,x,y)).T
                det = np.c_[(np.abs(diff) > tol).astype(int)]

                return (det,data)

        def CrossValidate(self,c_range=np.logspace(-3.0,3.0,7),gamma_range=np.logspace(-3.0,3.0,7)):
                param_grid = dict(C=c_range,gamma=gamma_range)
                cv = StratifiedShuffleSplit(self.det,n_iter=5,test_size=0.2)
                grid = GridSearchCV(svc(kernel='rbf',cache_size=1000),param_grid=param_grid,cv=cv)
                scale_mag,scale_x,scale_y = self.Scale(self.data)

                if self.spatial:
                        grid.fit(np.c_[scale_mag,scale_x,scale_y],self.det.ravel())

                else:
                        grid.fit(np.c_[scale_mag],self.det.ravel())

                print("The best parameters are %s with a score of %0.2f"
                          % (grid.best_params_, grid.best_score_))

        def predict_proba(self):
                '''
                Wrapper function to call predict probability on the SVM.
                Allows either full object to be saved, or part of object.
                '''
                return self.complete

        def save(self,filename,full=True):
                '''
                Save current state of the object

                If full is false, only save the SVM
                '''
                if full:
                        #save the entire object, including data
                        pickle.dump(self,open(filename,'wb'),protocol=-1)

                else:
                        #only save the SVM object
                        pickle.dump(self.completeness,open(filename,'wb'),protocol=-1)

        def calc_norm(self,grid=np.linspace(18.,26.,10000),fg_faint=26.,fg_bright=18.):
                '''
                Calculate p_cont(obs) for the completeness object. Must be run first, then can run
                calc_norm_gc in MCMC step to calculate p_gc(obs|theta), which depends on theta.

                Inputs: grid, np.array
                        grid over which to perform riemann sum to evaluate quantity

                        fg_faint: magnitude down to which we consider contamination
                        fg_bright: magnitude up to which we consider contamination

                #Evaluate the completeness on a grid, store for normalization of
                #GC completeness later
                self.complete_grid = np.exp(self.QueryComplete(grid)[:,0])
                self.faint_complete_ind= np.argmin(np.abs(self.complete_grid - 0.01))
                self.bright_complete_ind =  np.argmin(np.abs(self.complete_grid - .999))
                self.mag_grid_delta = (max_mag - min_mag) / grid_size

                #Calculate the completeness normalization for the contaminants
                #right now
                self.faint_complete_mag = self.mag_grid[self.faint_complete_ind]
                self.bright_complete_mag = self.mag_grid[self.bright_complete_ind]

                #check if we need to include both parts of the FG normalization integral,
                #or if we haven't populated down to this completeness
                if self.fg_faint > self.faint_complete_mag:
                        self.fg_complete_norm = 1./(self.faint_complete_mag-self.fg_bright) * ((self.bright_complete_mag - self.fg_bright)\
                                + np.sum(self.complete_grid[self.bright_complete_ind:self.faint_complete_ind]) * self.mag_grid_delta)

                elif (self.fg_faint < self.faint_complete_mag) and (self.fg_faint > self.bright_complete_mag):
                        #need to find the location of the fixed fg limit in the completeness function
                        self.fg_faint_ind = np.argmin(np.abs(self.mag_grid - self.fg_faint))

                        self.fg_complete_norm = 1./(self.fg_faint - self.fg_bright) * ((self.bright_complete_mag-self.fg_bright)\
                        + np.sum(self.complete_grid[self.bright_complete_ind:self.fg_faint_ind]) * self.mag_grid_delta)

                                #if we only populated above the completeness function, then we observed everything
                                else:
                                        self.fg_complete_norm = 1.
                        else:
                                self.faint_complete_mag = self.fg_faint

                return self.fg_complete_norm

        #def calc_gc_norm(self):
        '''



class CompleteNormalization(object):
        '''
        Object to calculate p(obs|theta), i.e. normalization of the completeness correction.

        Workflow:
                First, need to calculate marginal completeness of colors when given
                completeness as a function of brightness.

                Next, need to calculate completeness normalization for contamination,
                requires numerical integration over the contaminant color and magnitude
                distributions.

                Next, need to call completeness normalization for GCs as part of MCMC loop, needs to be fast.
                Contamination normalization must be ran first.


        Inputs:
                gc_density object
                fg_density object
                complete: array of completeness objects for each filter

                filters: int
                        number of filters being considered

                faint : array-like
                        faintest magnitude in various filters that we consider in the problem (WRT density estimation)

                bright : array-like
                        brightest mag in various filters that we consider in the problem

                c_blue: array-like
                        array of blue color limits

                c_red: array-like
                        array of red color limits

        '''
        def __init__(self,gc_density,fg_density,complete,faint=[26.,26.,26.],bright=[18.,18.,18.],filters=1,c_blue=[0.0,0.0],\
                                c_red=[2.0,1.5],fixed_cov=True,fixed_lum=True):
                #store the objects for density calculations
                self.gc_density = gc_density
                self.fg_density = fg_density
                self.complete = complete

                #store the magnitude limits for the problem
                self.faint = faint
                self.bright = bright

                self.filters = filters
                self.fixed_cov = fixed_cov

                if self.filters > 1:
                        #store the color limits for integration, if considering multiple filters
                        self.c_blue = c_blue
                        self.c_red = c_red


        def marginalize_color_complete(self,grid_1,grid_2,color=[0.],faint_mag=26.,bright_mag=18.):
                '''
                Calculate p(obs|m_1 - m_2) given p(obs|m_1),p(obs|m_2), and a fixed color

                color: array-like
                grid_1,2: completeness grids (2d arrays with mag,completeness) which span relevant ranges

                '''
                norm = []
                for c in color:
                        #first, need to calculate number of indicies to subtract off of m_2
                        mag_grid_spacing =  grid_1[0,1] - grid_1[0,0]
                        ind_offset = np.round(c / mag_grid_spacing)

                        if ind_offset < 0:
                                ind_offset = np.abs(ind_offset)
                                new_mag_grid = grid_1[0,:-ind_offset]
                                new_complete_grid = grid_1[1,:-ind_offset] * grid_2[1,ind_offset:]

                        elif ind_offset == 0:
                                new_mag_grid = grid_1[0,:]
                                new_complete_grid = grid_1[1,:]*grid_1[1,:]

                        #create new grids offset by appropriate amounts and calc completeness product
                        else:
                                new_mag_grid = grid_1[0,ind_offset:]
                                new_complete_grid = grid_1[1,ind_offset:] * grid_2[1,:-ind_offset]

                        #find indicies where bright and faint magnitudes occur
                        faint_m1_ind = np.argmin(np.abs(new_mag_grid - faint_mag))
                        bright_m1_ind =  np.argmin(np.abs(new_mag_grid - bright_mag))

                        #marginalize over the magnitude
                        norm.append(np.sum(new_complete_grid[bright_m1_ind:faint_m1_ind] * mag_grid_spacing / (faint_mag - bright_mag)))

                color_complete_grid = np.stack([color,norm],axis=1)

                return color_complete_grid

        def calc_fg_norm_color(self,c1_grid_res=0.01,c2_grid_res=0.01):
                '''
                Calculate p(obs), i.e. normalization, for the color distribution.

                '''
                #marginalzie over the completeness curves to get p(obs|color)
                self.c1 = np.arange(self.c_blue[0],self.c_red[0],c1_grid_res)
                self.c2 = np.arange(self.c_blue[1],self.c_red[1],c2_grid_res)
                self.color_grid_1 = self.marginalize_color_complete(
                                self.large_complete[0],self.large_complete[2],color=self.c1,faint_mag=self.faint[0],bright_mag=self.bright[0])
                self.color_grid_2 = self.marginalize_color_complete(
                                self.large_complete[1],self.large_complete[2],color=self.c2,faint_mag=self.faint[1],bright_mag=self.bright[1])

                #perform the 2D sum to calculate normalization term for color
                cc1,cc2 = np.meshgrid(self.c1,self.c2)
                inds_1,inds_2 = np.meshgrid(np.linspace(0,self.c1.size-1,self.c1.size),np.linspace(0,self.c2.size-1,self.c2.size))
                grid_size_2d = c1_grid_res * c2_grid_res
                grid_area = (self.c_red[0] - self.c_blue[0]) * (self.c_red[1]- self.c_blue[1])

                col_like = np.exp(self.fg_density.lnLikeColor(cc1.ravel(),cc2.ravel()))
                col_like_norm = 1./(np.sum(col_like)*grid_size_2d)

                color_norm = np.sum(col_like*col_like_norm*\
                        self.color_grid_1[inds_1.ravel().astype(int)][:,1]*\
                        self.color_grid_2[inds_2.ravel().astype(int)][:,1]*\
                        grid_size_2d)

                return color_norm

        def calc_fg_norm_lum(self,complete_grid):
                '''
                Calculate p(obs) normalization for the luminosity distribution,
                currently assumed d_fg = constant

                '''
                grid_res = complete_grid[0,1] - complete_grid[0,0]

                bright_mag_ind = np.argmin(np.abs(complete_grid[0,:] - self.bright[0]))
                faint_mag_ind = np.argmin(np.abs(complete_grid[0,:] - self.faint[0]))

                grid_norm = 1./(self.faint[0]-self.bright[0])

                self.fg_norm = grid_norm

                return np.sum(complete_grid[1,bright_mag_ind:faint_mag_ind] * grid_norm * grid_res)

        def calc_fg_norm(self):
                '''
                Calc the full p(obs) for the FG sample
                '''
                #first, need to make large completeness grids
                self.large_complete = []
                for i in range(self.filters):
                        self.large_complete.append(self.make_complete_grid(
                                self.complete[i],test_bright_mag=self.bright[i],test_faint_mag=self.faint[i]))

                if self.filters > 1:
                        color_norm = self.calc_fg_norm_color()

                else:
                        color_norm = 1.0

                lum_norm = self.calc_fg_norm_lum(self.large_complete[0])

                return lum_norm * color_norm


        def calc_gc_norm(self,means,cov,lum_means,lum_sigs,fractions):
                '''
                Calc the full p(obs|theta) for the GC distributions.

                Can only be run after calc_fg_norm() has already been ran.
                '''
                if self.filters > 1:
                        color_norm = self.calc_gc_norm_color(cov,means,fractions)

                else:
                        color_norm = 1.0

                lum_norm = self.calc_gc_norm_lum(lum_means,lum_sigs)

                return lum_norm * color_norm

        def calc_gc_norm_color(self,cov,means,fractions):
                '''
                Calculate the full p(obs|theta) for the GC color distribution.
                '''
                c1_grid_res = self.c1[1] - self.c1[0]
                c2_grid_res = self.c2[1] - self.c2[0]
                cc1,cc2 = np.meshgrid(self.c1,self.c2)
                inds_1,inds_2 = np.meshgrid(np.linspace(0,self.c1.size-1,self.c1.size),np.linspace(0,self.c2.size-1,self.c2.size))
                grid_size_2d = c1_grid_res * c2_grid_res
                grid_area = (self.c_red[0] - self.c_blue[0]) * (self.c_red[1]- self.c_blue[1])

                if self.fixed_cov:
                        col_like = np.exp(self.gc_density.lnLikeColor(np.stack([cc1.ravel(),cc2.ravel()],axis=1),fractions=fractions))

                else:
                        col_like = np.exp(self.gc_density.lnLikeColor(np.stack([cc1.ravel(),cc2.ravel()],axis=1),cov=cov,means=means,fractions=fractions))

                #col_like_norm = 1./(np.sum(col_like)*grid_size_2d)
                #norm_color_like = np.sum(fractions) / color_like
                col_like_norm = 1.

                color_norm = np.sum(col_like*col_like_norm*\
                        self.color_grid_1[inds_1.ravel().astype(int)][:,1]*\
                        self.color_grid_2[inds_2.ravel().astype(int)][:,1]*\
                        grid_size_2d)

                return color_norm

        def calc_gc_norm_lum(self,lum_means,lum_sigs):
                '''
                Calculate the full p(obs|theta) for the gc luminosity distribution.

                Currently only works for single luminosity function.
                '''
                complete_grid = self.large_complete[0]
                bright_mag_ind = np.argmin(np.abs(complete_grid[0,:] - self.bright[0]))
                faint_mag_ind = np.argmin(np.abs(complete_grid[0,:] - self.faint[0]))

                grid_res = complete_grid[0,1] - complete_grid[0,0]
                test_grid = complete_grid[0,bright_mag_ind:faint_mag_ind]

                #calculate probability from bright tail of luminosity distribution
                #bright_prob = stats.norm.cdf(bright[0],loc=lum_means,sig=lum_sigs)
                lum_like = np.exp(self.gc_density.lnLikeMag(test_grid,lum_means=lum_means,lum_sigs=lum_sigs))

                #we need to normalize lum_like to account for the fact that the gaussian distribution
                #extends beyond the bright and faint magnitudes.
                #lum_norm = 1. /(np.sum(lum_like) * grid_res)

                self.test_grid = test_grid
                self.lum_like = lum_like
                #self.lum_norm = lum_norm
                self.test_complete = complete_grid[1,bright_mag_ind:faint_mag_ind]

                return np.sum(complete_grid[1,bright_mag_ind:faint_mag_ind] * lum_like * grid_res)


        def make_complete_grid(self,complete,test_bright_mag=22.,test_faint_mag=26.,bright_mag=0.,faint_mag=40.,grid_size=0.002):
                '''
                Quick function to create a useful, stupidly big 2d completeness array,
                test_bright_mag: brightest value to actually query complete on
                test_faint_mag: faintiest value to actually query complete on
                bright_mag: value to append up 1's to
                faint_mag: value to append 0's down to

                grid_size=resolution on which to make the grid
                '''
                test_grid = np.arange(test_bright_mag,test_faint_mag,grid_size)

                complete_grid = np.exp(complete.QueryComplete(test_grid)[:,0])
                mag_array_bright = np.sort(np.arange(bright_mag,test_bright_mag,grid_size))
                complete_array_bright = np.ones(mag_array_bright.size)

                mag_array_faint = np.sort(np.arange(test_faint_mag,faint_mag,grid_size))
                complete_array_faint = np.zeros(mag_array_faint.size)

                full_complete = np.concatenate([complete_array_bright,complete_grid,complete_array_faint])
                full_mags = np.concatenate([mag_array_bright,test_grid,mag_array_faint])

                full_array = np.stack([full_mags,full_complete],axis=0)

                return full_array




















