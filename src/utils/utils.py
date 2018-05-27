import tensorflow as tf
from scipy.misc import imresize, imshow
import numpy as np
import scipy.signal
import os
from torch.autograd import Variable
import torch

def remove_none_grads(grads, vars):
    '''
    Replaces None with zeros of same shape as var
    :param grads: list of gradients
    :param vars: list of variables
    :return: gradients
    '''
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(vars, grads)]

def combine_gradients(grads_1, grads_2, variables):
    '''
    Adds gradients
    :param grads_actor: gradients for actor
    :param grads_critic: gradients for critic
    :param vars: trainable variables
    :return: gradients
    '''
    grads_1 = remove_none_grads(grads_1, variables)
    grads_2 = remove_none_grads(grads_2, variables)
    for i in range(len(grads_1)):
        grads_1[i] += grads_2[i]
    return grads_1

def process_image(img, img_h, img_w, img_c):
    ''' Crops, normalize, and reshapes the image to 84x84x3 '''
    img = img[35:195] # crop
    img = imresize(img, [img_h, img_w, img_c]) / 255 - .5 # resize and "kinda" normalize
    img = np.reshape(img, [img_c, img_h, img_w])
    return img

def advantage_with_terminals(R, terminal, state_values, gamma=.99):
    ''' computes discounted returns with terminal states as masks '''
    Gt, adv =[], []
    Gt.append( R[-1] + gamma * state_values[-1].data * (1 - terminal[-1]) )
    adv.append( Gt[-1] - state_values[-2].data)
    for i in reversed(range(len(R) - 1)):
        Gt.append(R[i] + gamma * Gt[-1] * (1- terminal[i]))
        adv.append( Gt[-1] - state_values[i].data)
    adv.reverse()
    Gt.reverse()
    return adv, Gt

def generalized_adv_with_terminals(R, terminal, state_values, gamma=.99, lmbda=1.):
    ''' calculates the advantage used in A3C (with state dependencies) and the 
        generalized advantage estimation
    '''
    adv,  gae = [], []
    Gt =  state_values[-1].data * (1 - terminal[-1]) 
    gae.append(0)
    for i in reversed(range(len(R))):
        Gt = R[i] + gamma * Gt * (1- terminal[i])
        adv.append( Variable(Gt) - state_values[i])
        td_res = R[i] + gamma* state_values[i+1].data * (1 - terminal[i]) - state_values[i].data
        gae.append(gae[-1] * gamma * lmbda * (1- terminal[i]) + td_res)
    adv.reverse()
    gae.reverse()
    return adv, gae[:-1]

def discount(R, gamma=.99):
    """
    Compute discounted sum of future values
    out[i] = in[] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return np.array(scipy.signal.lfilter([1], [1, -gamma], R[::-1], axis=0)[::-1])

def set_ckpt_flag(worker_id, ep_num, save_freq=500):
    return worker_id==0 and ep_num%save_freq==0
        
def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def kl_divergence_cat(p, q):
    ''' Calculates the KL-divergence betwen two categorical i.e. KL(p||q)'''
    kl = p * torch.log(p / q)
    kl = torch.sum(kl, 1, keepdim=True)
    return kl

def trpo_scaling(k, g, delta=1):
    foo = (k.transpose() * g - delta) / torch.norm(k, 2)
    max_term = torch.max(0, foo)
    return g - max_term * k