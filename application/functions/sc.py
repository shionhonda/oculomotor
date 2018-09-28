import numpy as np
import brica
from .pfc import Phase

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
        self.search_direction = 0

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')

        # Likelihoods and eye movment params from accumulators in FEF module.
        fef_data = inputs['from_fef'][0]
        phase = inputs['from_fef'][1]
        angle = inputs['from_fef'][2] # (2)
        # Likelihood thresolds from BG module.
        bg_data = np.clip(inputs['from_bg'][:10], 0, 8)
        phase = inputs['from_bg'][10]

        action = self._decide_action(phase, fef_data, bg_data, angle)
        
        # Store FEF data for debug visualizer
        self.last_fef_data = fef_data
        
        return dict(to_environment=action)

    def _decide_action(self, phase, fef_data, bg_data, angle):
        '''
          Function: Choose the direction with maximum likelihood after addition of baseline
          Inputs: 
            bg_data: saliency baseline
        '''
        self.baseline = gauss_mixture(bg_data)
        distrib = fef_data[:,0]+self.baseline
        self.last_sc_data = distrib
        max_idx = np.argmax(distrib)
        if phase==Phase.SEARCH:
            action = fef_data[max_idx, 1:]
        elif phase==Phase.EXPLORE:
            x,y = angle[0], angle[1]
            explore_idx = self.explore(x, y)
            # Sometimes choose point with high saliency
            p = [np.max(distrib), 0.4]/(np.max(distrib)+0.4)
            idx = np.random.choice([max_idx, explore_idx], p=p)
            action = fef_data[idx, 1:]
        else:
            action = fef_data[27, 1:]
        return action

    def explore(self, x, y):
        dirs = [[0.21,0.21], [-0.21,0.21], [-0.21,-0.21], [0.2,-0.21]]
        dx = dirs[self.search_direction][0] - x
        dy = dirs[self.search_direction][1] - y
        th = 0.02
        if dx>th and dy>th:
            return 18
        elif dx>th and dy<-th:
            return 21
        elif dx<-th and dy>th:
            return 42
        elif dx<-th and dy<-th:
            return 45
        elif dx>th:
            return np.random.choice([19,20])
        elif dy>th:
            return np.random.choice([26,34])
        elif dx<-th:
            return np.random.choice([43,44])
        elif dy<-th:
            return np.random.choice([29,37])
        else:
            self.search_direction = (self.search_direction+1)%4
            return np.random.choice([27,28,35,36])


def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)

def gauss_mixture(params):
    '''
      Function: Make mixture of 2 2D gaussian distribution according to the input parameters
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