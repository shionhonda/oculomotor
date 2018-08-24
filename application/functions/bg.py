import brica
import numpy as np
import chainer
from chainer import optimizers
import chainerrl

class BG(object):
    def __init__(self, alpha=0.5, gamma=0.95):
        self.timing = brica.Timing(5, 1, 0)
        self.agent = _set_agent()
        self.reward = 0
        chainer.config.train = False
        chainer.config.enable_backprop = False
        

    def __call__(self, inputs):
        """
        Function:
          Define likelihood thresholds for SC using DDPG.
          Note that this model only uses saliency map for testing.
        Arguments:
          inputs['from_fef']: list size of 128*3, that is, 8*8*2*[likelihood, ex, ey].
          saliency accumulation, cursor accumulation in order.
        Outputs:
          likelihood_thresholds: numpy array size of 64
        """
        # from_envがない

        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')

        fef_data = np.array(inputs['from_fef'])
        state = fef_data[:, 0]
        action = self.agent.act_and_train(state, self.reward)
        reward, done = inputs['from_environment']

        return dict(to_pfc=None, to_fef=None, to_sc=action)

def _phi(obs):
    return obs.astype(np.float32)

def _set_agent(actor_lr=1e-4, critic_lr=1e-3, gamma=0.995, minibatch_size=200):
    q_func = chainerrl.q_functions.FCSAQFunction(
        128, 129,
        n_hidden_channels=256,
        n_hidden_layers=3)
    pi = chainerrl.policy.FCDeterministicPolicy(
        128, action_size=129,
        n_hidden_channels=256,
        n_hidden_layers=3,
        min_action=0, max_action=1,
        bound_action=True)
    model = chainerrl.agents.ddpg.DDPGModel(q_func=q_func, policy=pi)
    opt_a = optimizers.Adam(alpha=actor_lr)
    opt_c = optimizers.Adam(alpha=critic_lr)
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])
    opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
    opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')
    rbuf = chainerrl.replay_buffer.ReplayBuffer(5 * 10 ** 5)
    ou_sigma = 1 * 0.2 # Action space width
    explorer = chainerrl.explorers.AdditiveOU(sigma=ou_sigma)

    agent = chainerrl.agents.ddpg.DDPG(model, opt_a, opt_c, rbuf, gamma=gamma,
                explorer=explorer, replay_start_size=5000,
                target_update_method='soft',
                target_update_interval=1,
                update_interval=4,
                soft_update_tau=1e-2,
                n_times_update=1,
                phi=_phi, gpu=-1, minibatch_size=minibatch_size)
    return agent