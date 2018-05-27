import gym
import numpy
from multiprocessing import Process, Pipe
import numpy as np
from src.utils.utils import process_image

def worker(remote, env, id):
    """
    worker 
    :param remote: remote end of pipe 
    :param env: environment
    """
    render = True if id==0 else False
    episodic_reward = 0
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, _ = env.step(data)
            episodic_reward += reward
            info = False
            if done:
                info = episodic_reward
                episodic_reward = 0       
                ob = env.reset()            
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_action_space':
            remote.send(env.action_space)
        else:
            raise NotImplementedError
        if render:
            env.render()

class AtariEnvWrapper(gym.Wrapper):
    """
    Wrapper for the Atari environments
    """
    def __init__(self, env, img_h, img_w, img_c):                
        
        gym.Wrapper.__init__(self, env)
        print('created wrapper')
        self._img_h = img_h
        self._img_w = img_w
        self._img_c = img_c
        

    def _reset(self):
        """
        Resets the environment
        """   
        self.env.reset()
        obs, _, _, _ = self.env.step(2)
        obs = process_image(obs, self._img_h, self._img_w, self._img_c)        
        return obs

    def _step(self, action):
        """
        Takes one step in environment according to the current action
        :param self: 
        :param action: action to take 
        return: observation, reward, terminal, info
        """

        obs, r, t, info = self.env.step(action)
        if t:
            obs = self._reset()
        else:
            obs = process_image(obs, self._img_h, self._img_w, self._img_c)
        return obs, r, t, info
    
class VectorizedEnvironment(object):
    """
    creates and contains a number of environments
    """
    
    def __init__(self, num_envs, env_id, img_h=84, img_w=84, img_c=3):
        """
        docstring here
        :param self: 
        :param num_envs: number of environments to use 
        :param env_id: id of environment, e.g. 'Breakout-v0'
        :param img_h=84: height of input image
        :param img_w=84: width of input image
        :param img_c=3: number of color channels for input image
        """   

        # setup pipes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])     

        # setup processes
        self.processes = [Process(target=worker, args=(work_remote,  AtariEnvWrapper(gym.make(env_id ), img_h, img_w, img_c), id))  \
                         for id, work_remote in enumerate(self.work_remotes)]
        
        # start processes
        for process in self.processes:
            process.start()
        
        # get action space
        self.remotes[0].send(('get_action_space', None))
        self._action_space = self.remotes[0].recv()
        
    def step(self, actions):
        """
        Take one step in all environments
        :param self: 
        :param actions: actions to performe in environments
        """   
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)        
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
        resets all environments        
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        """
        close all environments
        """ 
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
            
    @property
    def num_envs(self):
        return len(self.remotes)

    @property
    def action_space(self): 
        return self._action_space