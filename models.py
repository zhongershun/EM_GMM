import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from utils import get_random_psd
import configs
from matplotlib import pyplot as plt
class GMM2d():
    def __init__(self, params={}):
        self.n_components = 2
        self.params = params if params else self.initialize_random_params()
    
    def initialize_random_params(self):
        np.random.seed(0)  ## 定义一个固定的随机种子方便后续实验
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
        # print(self.params)

    def e_step(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        pdf = super().get_pdf(x)
        ## pdf[:,0] -- p(x|y=0;theta)p(y=0;theta)
        ## pdf[:,1] -- p(x|y=1;theta)p(y=1;theta)
        # print("\npdf ",pdf)

        pdf = np.array(pdf)
        # print("\npdf.shape ",pdf.shape)
        
        Q_y0 = pdf[:,0]/(pdf[:,0]+pdf[:,1])  ## 记录每一个样本中y的取值为0的概率
        # print("\nQ_y1 ",Q_y1)
        # print("\nQ_y1.shape ",Q_y0.shape)
        # print(np.max(Q_y0))
        
        Q_y1 = 1 - Q_y0                      ## 记录每一个样本中y的取值为1的概率  
        
        logLH = np.mean(np.log(pdf.sum(axis=1)))  ## 计算极大似然估计

        pass

        return Q_y0, Q_y1, logLH

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    def m_step(self, x, Q_y0, Q_y1):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Q_y0, Q_y1 = self.e_step(x)
        
        N = x.shape[0] ## 样本数量
        
        phi = np.sum(Q_y1)/N
        
        # print("phi ",phi)
        # print("real phi ",self.params['phi'])

        Q_y0_re = Q_y0.reshape(N,1)
        Q_y1_re = Q_y1.reshape(N,1)
        mu0 = np.sum(Q_y0_re*x,axis = 0)/np.sum(Q_y0)
        mu1 = np.sum(Q_y1_re*x,axis = 0)/np.sum(Q_y1)

        # print("mu0 ",mu0)
        # print("real mu0 ",self.params['mu0'])
        # print("mu1 ",mu1)
        # print("real mu1 ",self.params['mu1'])

        sqrt_q_y0 = np.sqrt(Q_y0).reshape(N,1)
        sqrt_q_y1 = np.sqrt(Q_y1).reshape(N,1)
        x_mu0 = x - mu0
        x_mu1 = x - mu1
        x_mu0 = x_mu0*sqrt_q_y0
        x_mu1 = x_mu1*sqrt_q_y1
        x_mu0_T = x_mu0.T
        x_mu1_T = x_mu1.T
        sigma0 = x_mu0_T.dot(x_mu0)/np.sum(Q_y0)
        sigma1 = x_mu1_T.dot(x_mu1)/np.sum(Q_y1)
        # print(sigma0)
        # print(sigma1)
        # print( ((x[0,:]-mu0).T).dot(x[0,:]-mu0))
        # for i in range(N):
            
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
        return self.params

    def run_em(self,x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        echo_times = 30
        logLHs = []
        Q_y0 = Q_y1 = []
        predict_proba = []
        for _ in range(echo_times):
            Q_y0, Q_y1, logLH = self.e_step(x)
            logLHs.append(logLH)
            self.m_step(x,Q_y0,Q_y1)
        # print(self.params)
        predict_proba.append(Q_y0)
        predict_proba.append(Q_y1)
        predict_proba = np.array(predict_proba)
        predict_proba = predict_proba.T
        # print("predict_proba.shape ",predict_proba.shape)
        # print(Q_y0)
        plt.figure(figsize=(8,5))
        x_axis = [i for i in range(echo_times)]
        plt.plot(x_axis, logLHs, label="line L", color='lime', alpha=0.8, linewidth=2, linestyle="--")
        plt.show()
        
        mask = Q_y1 > 0.5 ## 将多次迭代后y取1的概率大于0.5的样本预测为类别1
        forecast = 1*mask

        pass
        return forecast, predict_proba

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
