import argparse
import numpy as np
from src.utils.vectorized_environment import VectorizedEnvironment
from src.model.a2c import A2cModel
from src.utils.utils import generalized_adv_with_terminals, maybe_create_dir, set_ckpt_flag
from src.utils.data_logger import DataLogger
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch

class EnvRunner(object):
    
    def __init__(self, gamma, model, env, num_workers,  img_h, img_w, img_c, logger):
        self.gamma = gamma
        self.env = env
        self.model = model
        self.obs = torch.from_numpy(env.reset()).float()        
        self.img_h, self.img_w, self.img_c = img_h, img_w, img_c
        self.num_workers = num_workers
        self.ep_counter = np.zeros(num_workers)
        self.logger = logger

    def run_seq(self, num_steps, lstm_state):
        """
        follow the current policy for a fixed sequence
        :param num_steps: length of sequence
        :param lstm_state: current state of rthe lstm-state for all workers
        """   
        # observation is 1 elm longer due to that we need that one for bootstraping
        reward_buff = []
        terminal_buff = []
        state_value_buff = []
        entropy_buff = []
        log_prob_buff = []
        hx, cx = lstm_state
        hx, cx = Variable(hx), Variable(cx)
        ckpt_flag = False
        for i in range(num_steps):
            
            value, log_prob, actions, entropy, lstm_state = self.model.forward(Variable(self.obs).cuda(), hx, cx)            
            hx, cx = lstm_state
            new_obs, r, terminal, info = self.env.step(actions.squeeze())

            # if episode is finished, reset hidden states and store logs
            for worker_id in np.where(info)[0]:                
                self.ep_counter[worker_id] += 1
                hx[worker_id, :].data.fill_(0)
                cx[worker_id, :].data.fill_(0)
                print('|Worker: ', worker_id, '|Episode: ', self.ep_counter[worker_id], 
                      '|Reward: ', info[worker_id], ' | Value: ', value.cpu().data.numpy().mean() )
                self.logger.store_data(worker_id, self.ep_counter[worker_id], info[worker_id])
                ckpt_flag = set_ckpt_flag(worker_id, self.ep_counter[worker_id], 1)

            r = np.expand_dims(r, axis=1)
            r = torch.from_numpy(r).float().cuda()
            reward_buff.append(r)
            entropy_buff.append(entropy)            
            log_prob_buff.append(log_prob)
            terminal = [[ float(t) ] for t in terminal]
            terminal = torch.FloatTensor(terminal).cuda()
            terminal_buff.append(terminal)
            state_value_buff.append(value)
            self.obs = torch.from_numpy(new_obs).float()
            
        hx, cx = lstm_state
        value, _, _, _, _ = self.model.forward(Variable(self.obs).cuda(), hx, cx)            
        state_value_buff.append(value) # add last state-value for bootstraping
        return log_prob_buff, reward_buff, terminal_buff, state_value_buff, entropy_buff, hx.data, cx.data, ckpt_flag
    
def train(env_runner, model, hidden_size, num_workers, lr, entropy_reg, ckpt_dir, horizon=20, gamma=.99):
    """
    training loop for all workers
    :param env_runner: EnvRunner object containing all workers and environments
    :param model: our model
    :param horizon=20: num steps to take in environment before training
    :param gamma=.99: discount factor
    """
    cx = torch.zeros(num_workers, hidden_size).cuda()
    hx = torch.zeros(num_workers, hidden_size).cuda()
    ep_reward, mean_state_value = 0, 0
    print('LR', lr)
    optimizer = optim.Adam(model.parameters() , lr)
    ckpt_ticker = 1
    while "It ain't over till it's over":
         # lstm state at the start of the sequence
        lstm_state = (hx, cx)
        log_probs, rewards, terminals, state_values, entropy, hx, cx, ckpt_flag = env_runner.run_seq(horizon, lstm_state)
        
        if ckpt_flag:
            store_dict =  {'model':model.state_dict(), 'optimizer':optimizer.state_dict}
            filename =  ckpt_dir + '/checkpoint_' + str(ckpt_ticker) + '.pth.tar' 
            torch.save(store_dict, filename)
            ckpt_ticker +=1

        ''' calculate gradients and backprop '''
        actor_loss, critic_loss  = 0, 0 
        adv, gae = generalized_adv_with_terminals(rewards, terminals,  state_values, gamma)
        for i in range(len(log_probs)):
            critic_loss += .25 * adv[i].pow(2)
            actor_loss -= log_probs[i] * Variable(gae[i]) + entropy[i] * entropy_reg
        loss = (actor_loss + critic_loss).mean() # add mean loss for all workers               
        loss.backward() # calculate gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) # gradient clipping
        optimizer.step() # backprop
        optimizer.zero_grad()        
        
        del log_probs, rewards, terminals, state_values, entropy, adv
        
def main(args):
    """
    docstring here
    :param lr: learning rate
    :param num_workers: number of workers
    :param env: environment id
    :param img_h: height of input images
    :param img_w: width of input images
    :param img_c: number of channels for images
    :param gamma: discount factor
    """
    maybe_create_dir(args.log_dir)
    maybe_create_dir(args.ckpt_dir)
    logger = DataLogger(args.log_dir)
    env = VectorizedEnvironment(args.workers, args.env, args.img_h, args.img_w, args.img_c)    
    num_actions = env.action_space.n
    model = A2cModel(args.hidden, args.img_h, args.img_w, args.img_c, num_actions)
    model.cuda()
    env_runner = EnvRunner(args.gamma, model, env, args.workers, args.img_h, args.img_w, args.img_c, logger)
    train(env_runner, model, args.hidden, args.workers, args.lr, args.ent_reg, args.ckpt_dir, args.horizon, args.gamma)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate for model', type=float, default=1e-4)
    parser.add_argument('--gamma', help='discount factor', type=float, default=.99)
    parser.add_argument('--horizon', help='number of steps before bootstrap', type=float, default=20)
    parser.add_argument('--ent_reg', help='entropy regularizer factor', type=float, default=1e-2)
    parser.add_argument('--hidden', help='size of hidden state for LSTM', type=float, default=512)
    parser.add_argument('--workers', help='number of workers to explore the environment', type=int, default=16)
    parser.add_argument('--env', help='environment ID', default='Breakout-v4')
    parser.add_argument('--img_h', help='height of input image', type=int, default=84)
    parser.add_argument('--img_w', help='height of input image', type=int, default=84)
    parser.add_argument('--img_c', help='number of color channels for input images', type=int, default=3)
    parser.add_argument('--log_dir', help='log directory', default='./a2c_logs')
    parser.add_argument('--ckpt_dir', help='checkpoint directory', default='./a2c_checkpoints')
    args = parser.parse_args()  
    main(args)