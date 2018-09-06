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
        self.thresholds = None

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')

        # Likelihoods and eye movment params from accumulators in FEF module.
        fef_data = inputs['from_fef']
        # Likelihood thresolds from BG module.
        bg_data = inputs['from_bg']

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
        self.thresholds = bg_data[:-1]
        diff = fef_data[:64,0]-bg_data[:-1]
        self.last_sc_data = diff
        max_idx = np.argmax(self.last_sc_data)
        action = fef_data[max_idx, 1:]
        return action

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)