import argparse
import numpy
import brica
import chainer
from chainer import optimizers
from chainer.backends import cuda
import chainerrl
import time


"""
This is an example implemention of BG (Basal ganglia) module.
You can change this as you like.
"""

class BG(object):
    def __init__(self, gpuid=-1, alpha=0.5, gamma=0.95, train=True, backprop=True):
        self.timing = brica.Timing(5, 1, 0)
        self.agent = self._set_agent(gpuid=gpuid)
        self.reward = 0
        self.time = 0
        if gpuid<0:
            self.xp = numpy
        else:
            print("Use GPU")
            cuda.get_device(gpuid).use()
            self.xp = cuda.cupy

        chainer.config.train = train
        chainer.config.enable_backprop = backprop
        

    def __call__(self, inputs):
        """
        Function:
          Define likelihood thresholds for SC using DDPG.
          Note that this model only uses saliency map for testing.
        Arguments:
          inputs['from_fef']: list size of 128*3, that is, 8*8*2*[likelihood, ex, ey].
          saliency accumulation, cursor accumulation in order.
        Outputs:
          likelihood_thresholds: numpy array size of 128+1
        """

        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')
        if 'from_vc' not in inputs:
            raise Exception('BG did not recieve from VC')

        fef_data = self._phi(inputs['from_fef'][:])
        phase = inputs['from_pfc']
        penalty = inputs['from_vc']
        state = self._phi(fef_data[:, 0])
        action = self.agent.act_and_train(state, self.reward)
        reward, done = inputs['from_environment']
        self.time += 1
        if reward>0:
            self.time = 0
        self.reward = self.reward + reward + penalty
        output_sc = self.xp.hstack((action, phase))


        return dict(to_pfc=self.time, to_fef=None, to_sc=output_sc)

    def _phi(self, obs):
        return obs.astype(self.xp.float32)

    def _set_agent(self, gpuid=-1,actor_lr=1e-4, critic_lr=1e-3, gamma=0.995, minibatch_size=600):
        q_func = chainerrl.q_functions.FCSAQFunction(
            64, 10,
            n_hidden_channels=8,
            n_hidden_layers=3)
        pi = chainerrl.policy.FCDeterministicPolicy(
            64, action_size=10,
            n_hidden_channels=8,
            n_hidden_layers=3,
            min_action=0, max_action=8,
            bound_action=True)
        if gpuid>=0:
            q_func.to_gpu(gpuid)
            pi.to_gpu(gpuid)
        model = chainerrl.agents.ddpg.DDPGModel(q_func=q_func, policy=pi)
        if gpuid>=0:
            model.to_gpu()
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
                    phi=self._phi, gpu=gpuid, minibatch_size=minibatch_size)
        return agent
