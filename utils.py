import numpy as np

def get_random_psd(n): ## 生成协方差矩阵
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())

def learn_params(x_labeled, y_labeled):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = x_labeled.shape[0]
    y_labeled_re = y_labeled.reshape(n,1)
    
    phi = np.mean(y_labeled)
    I_y0 = 1-y_labeled_re
    I_y1 = y_labeled_re
    mu0 = np.sum(x_labeled*I_y0,axis=0)/np.sum(1-y_labeled)
    mu1 = np.sum(x_labeled*I_y1,axis=0)/np.sum(y_labeled)
    sigma0 = ((I_y0*(x_labeled-mu0)).T).dot((I_y0*(x_labeled-mu0)))
    sigma1 = ((I_y1*(x_labeled-mu1)).T).dot((I_y1*(x_labeled-mu1)))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    return params