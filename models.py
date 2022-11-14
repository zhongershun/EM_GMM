import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from utils import get_random_psd
import configs
class GMM2d():
    def __init__(self, params={}):
        self.n_components = 2
        self.params = params if params else self.initialize_random_params()
    
    def initialize_random_params(self):
        params = {'phi': np.random.uniform(0, 1),#phi corresponds to the probability of the second gaussian
              'mu0': np.random.normal(0, 1, size=(self.n_components,)),
              'mu1': np.random.normal(0, 1, size=(self.n_components,)),
              'sigma0': get_random_psd(self.n_components),
              'sigma1': get_random_psd(self.n_components)}
        return params
    
    def get_pdf(self,x):
        return np.array( [(1-self.params["phi"])*(stats.multivariate_normal(self.params["mu0"], self.params["sigma0"]).pdf(x)),\
            (self.params["phi"])*(stats.multivariate_normal(self.params["mu1"], self.params["sigma1"]).pdf(x))]).T
    
    def GMM_sklearn(self,x):
        model = GaussianMixture(n_components=2,
                                covariance_type='full',
                                tol=0.01,
                                max_iter=1000,
                                weights_init=[1-self.params['phi'],self.params['phi']],
                                means_init=[self.params['mu0'],self.params['mu1']],
                                precisions_init=[self.params['sigma0'],self.params['sigma1']])
        model.fit(x)
        print("\nscikit learn:\n\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
                % (model.weights_[1], model.means_[0, :], model.means_[1, :], model.covariances_[0, :], model.covariances_[1, :]))
        return model.predict(x), model.predict_proba(x)[:,1]


class EM(GMM2d):
    def __init__(self,params={}):
        super().__init__(params)
        print(self.params)

    def e_step(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        ## for i in range(self.n_components):
        Q_y1 = []  ## 记录每一个样本中y的取值为1的概率
        Q_y0 = []  ## 纪律每一个样本中y的取值为0的概率

        pdf = super().get_pdf(x)
        ## pdf[:,0] -- p(x|y=0;theta)
        ## pdf[:,1] -- p(x|y=1;theta)
        print("pdf.shape ",np.array(pdf).shape(0))
        print(pdf[:,0])  
        ## print("frac_0 ",pdf)

        # for i in range(self.n_components):
        #     frac_0 = 
        

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    def m_step(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params = {'phi': None, 'mu0': None, 'mu1': None, 'sigma0': None, 'sigma1': None}
        return self.params

    def run_em(self,x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
