import numpy as np
import brica


class SC(object):
    """ 
    SC (superior colliculus) module.
    SC outputs action for saccade eye movement.
    """
    def __init__(self):
        self.timing = brica.Timing(6, 1, 0)

        self.last_fef_data = None
        self.last_sc_data = None
        self.baseline = None

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')

        # Likelihoods and eye movment params from accumulators in FEF module.
        fef_data = inputs['from_fef']
        # Likelihood thresolds from BG module.
        bg_data = np.clip(inputs['from_bg'], 0, 8)

        action = self._decide_action(fef_data, bg_data)
        
        # Store FEF data for debug visualizer
        self.last_fef_data = fef_data
        
        return dict(to_environment=action)

    def _decide_action(self, fef_data, bg_data):
        '''
          Function: Choose the direction with maximum likelihood subtracted by threshold
          Inputs: 
            bg_data: 0-63 are saliency thresholds; 64-127 are cursor thresholds; 128 is lambda
        '''
        self.baseline = gauss_mixture(bg_data)
        diff = fef_data[:64,0]+self.baseline
        self.last_sc_data = diff
        max_idx = np.argmax(diff)
        action = fef_data[max_idx, 1:]
        return action

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)

def gauss_mixture(params):
    '''
      params: mu_x1, mu_y1, sigma_x1, sigma_y1, mu_x2, mu_y2, sigma_x2, sigma_y2, lamda1, lamda2
    '''
    mu1 = params[[0,1]]
    mu2 = params[[4,5]]
    det1 = (params[2]+0.1)*(params[3]+0.1)
    det2 = (params[6]+0.1)*(params[7]+0.1)
    inv1 = np.array([[1/(params[2]+0.1), 0], [0, 1/(params[3]+0.1)]])
    inv2 = np.array([[1/(params[6]+0.1), 0], [0, 1/(params[7]+0.1)]])
    lam1 = params[8]
    lam2 = params[9]

    def f(x, y):
        x_c1 = np.array([x, y]) - mu1
        exp1 = np.exp(- np.dot(np.dot(x_c1,inv1),x_c1[np.newaxis, :].T) / 2.0) 
        x_c2 = np.array([x, y]) - mu2
        exp2 = np.exp(- np.dot(np.dot(x_c2,inv2),x_c2[np.newaxis, :].T) / 2.0) 
        return lam1*exp1/(2*np.pi*np.sqrt(det1)) + lam2*exp2/(2*np.pi*np.sqrt(det2))

    x = y = np.arange(0,8)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(f)(X,Y)
    return Z.reshape(-1)