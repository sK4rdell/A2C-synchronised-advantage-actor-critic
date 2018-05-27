import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    if isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


class A2cModel(nn.Module):
    def __init__(self, num_hidden, img_h, img_w, img_c, num_actions):
        super(A2cModel, self).__init__()
        self._num_hidden = num_hidden
        self._num_actions = num_actions
        
        # declare layers
        self.conv1 = nn.Conv2d(img_c, 32, 5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTMCell(1024, self._num_hidden)
        self.affine_value = nn.Linear(self._num_hidden, 1)
        self.affine_policy = nn.Linear(self._num_hidden, self._num_actions)
        
        self.apply(weights_init)
        self.affine_value.weight.data = norm_col_init(self.affine_value.weight.data, 0.01)
        self.affine_value.bias.data.fill_(0)
        self.affine_policy.weight.data = norm_col_init(self.affine_policy.weight.data, 1.0)
        self.affine_policy.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
    def forward(self, obs, hx, cx):
        x = F.elu(self.max_pool1(self.conv1(obs)))
        x = F.elu(self.max_pool2(self.conv2(x)) )
        x = F.elu(self.max_pool3(self.conv3(x)))
        x = F.elu(self.max_pool4(self.conv4(x)))
        x = x.view(-1, 1024)

        hx, cx = self.lstm(x, (hx, cx))
        value = self.affine_value(hx)
        policy_logits = self.affine_policy(hx)  
        policy = F.softmax(policy_logits) # softmax policy
        actions = policy.multinomial().data # sample action from policy
        log_probs = F.log_softmax(policy_logits) # log pi(*|s)        
        log_prob = log_probs.gather(1, Variable(actions)) # log pi(*|s)
        entropy = - (log_probs * policy).sum(1, keepdim=True)             

        return value, log_prob, actions.cpu().numpy(), entropy, (hx, cx)

