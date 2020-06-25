import argparse
import gym
import gym_hanoi
import numpy as np
from itertools import count
from collections import namedtuple
from random import shuffle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

from PCFG import PCFG
from tqdm import tqdm

max_steps = 300
entropy_factor = 0.05
nt = 6
use_lstm = False
state_onehot = True

# Cart Pole
num_disks = 3
env_noise = 0.
pretrain = False
lr = 1e-3

if pretrain:
    print('using pretrained weights')
else:
    print('training from scratch')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make("Hanoi-v0")
env.set_env_parameters(num_disks, env_noise, verbose=True)

# env.seed(args.seed)
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

state_space_dim = num_disks
action_space_dim = 6



class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, vocab=action_space_dim,  # vocab = action_space
                 state_space=state_space_dim,
                 state_onehot = state_onehot,
                 nt_states=nt,
                 h_dim=32,
                 rule_dim=32,
                 z_dim=32,
                 state_dim=32,
                 use_lstm = use_lstm):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.rule_dim = rule_dim

        self.use_lstm = use_lstm

        self.state_onehot = state_onehot
        if state_onehot:
            self.state_space = state_space * 3
        else:
            self.state_space = state_space
        self.affine1 = nn.Linear(self.state_space, state_dim)

        if use_lstm:
            self.z_dim = z_dim
        else:
            self.z_dim = self.state_dim

        self.vocab = vocab
        self.nt_states = nt_states

        # action & reward buffer
        self.action_probs = []
        self.rewards = []
        self.hidden_state = None
        self.action_probs_hist = []

        self.leftmost = [0]   # leftmost unexanded non-terminal
        self.micro_execute = []  # micro action(s) to execute at this time step
        self.micro_storage = []  # rightmost terminal left unexecuted

        self.root_rule_num = nt_states * nt_states
        self.nt_rule_num = nt_states * (nt_states + vocab) * (nt_states + vocab)
        self.nt_nt_rule_num = (nt_states + vocab) * (nt_states + vocab)

        self.root_rule_emb = nn.Parameter(torch.randn(self.root_rule_num, rule_dim))
        self.nt_rule_emb = nn.Parameter(torch.randn(self.nt_rule_num, rule_dim))

        self.register_parameter('root_rule_emb', self.root_rule_emb)
        self.register_parameter('nt_rule_emb', self.nt_rule_emb)

        self.root_rule_mlp = nn.Sequential(nn.Linear(state_dim + self.z_dim, rule_dim),
                                      ResidualLayer(state_dim, rule_dim),
                                      ResidualLayer(state_dim, rule_dim),
                                      nn.Linear(state_dim, 1))    #self.root_rule_num))
        self.nt_rule_mlp = nn.Sequential(nn.Linear(state_dim + self.z_dim, rule_dim),
                                      ResidualLayer(state_dim, rule_dim),
                                      ResidualLayer(state_dim, rule_dim),
                                      nn.Linear(state_dim, 1))    #self.nt_rule_num))

        self.enc_rnn = nn.LSTM(state_dim, h_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.enc_params = nn.Linear(h_dim, z_dim * 2)

        self.state_changed = True

    def enc(self, state_emb):
        h, hidden_state = self.enc_rnn(state_emb.unsqueeze(0), self.hidden_state)
        self.hidden_state = hidden_state
        params = self.enc_params(h.max(1)[0])
        mean = params[:, :self.z_dim]
        logvar = params[:, self.z_dim:]
        return mean, logvar

    def forward(self, state, use_mean=True):

        assert len(self.micro_execute) == 0   # no unexecuted micro-actions

        if self.state_onehot:
            state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
            state = state_one_hot.scatter_(1, state, 1).view(1, -1).squeeze(0)  # 1 x (3 x state_space)
            state_emb = F.relu(self.affine1(state.unsqueeze(0)))

        if self.use_lstm:
            if self.state_changed:
                mean, logvar = self.enc(state_emb)

                batch_size = 1

                if use_mean:
                    z = mean
                else:
                    z = mean.new(batch_size, mean.size(1)).normal_(0, 1)
                    z = (0.5 * logvar).exp() * z + mean
                self.z = z
            else:
                z = self.z
        else:
            self.z = state_emb
            z = self.z

        nt = self.leftmost.pop(-1)
        assert nt == 0 or nt >= self.vocab

        if nt == 0:
            # calculate rule probs
            root_rule_emb = self.root_rule_emb   # root_rule_num x rule_dim
            z = z.expand(self.root_rule_num, self.z_dim)
            root_rule_emb = torch.cat([root_rule_emb, z], 1)
            root_rule_score = self.root_rule_mlp(root_rule_emb).squeeze()
            rule_prob = F.softmax(root_rule_score)  # root_rule_num x 1

            # select expansion rule
            m = Categorical(rule_prob)
            rule = m.sample()
            log_prob = m.log_prob(rule)

            nt1 = int(rule//self.nt_states + self.vocab)
            nt2 = int(rule % self.nt_states + self.vocab)

            self.leftmost = [nt1,nt2]
            self.rightmost = []
            self.mirco = []

        else:
            nt = nt - self.vocab
            z = z.expand(self.nt_nt_rule_num, self.z_dim)
            nt_rule_emb = self.nt_rule_emb   # nt_rule_num x rule_dim
            nt_rule_emb = nt_rule_emb.expand(self.nt_rule_num, self.rule_dim).view(self.nt_states,
                                                                              (self.vocab + self.nt_states)*(self.vocab + self.nt_states),
                                                                                           self.rule_dim)
            nt_rule_emb = nt_rule_emb[nt,:,:]  #  nt_nt_rules_num x rule_dim
            nt_rule_emb = torch.cat([nt_rule_emb, z], 1)
            nt_rule_score = self.nt_rule_mlp(nt_rule_emb).squeeze()
            rule_prob = F.softmax(nt_rule_score)

            # select expansion rule
            m = Categorical(rule_prob)
            rule = m.sample()
            log_prob = m.log_prob(rule)

            nt1 = int(rule//(self.vocab+self.nt_states))
            nt2 = int(rule%(self.vocab+self.nt_states))

            if nt2 < self.vocab:
                # nt2 is a micro-action
                if nt1 < self.vocab:
                    # nt1 is a micro-action, both micro
                    self.micro_execute = self.micro_storage
                    self.micro_storage = []
                    self.micro_execute.append(nt2)
                    self.micro_execute.append(nt1)
                else:
                    # nt1 is a macro-action
                    self.micro_storage.append(nt2)
                    self.leftmost.append(nt1)

            else:
                # nt2 is a macro-action
                self.leftmost.append(nt2)
                if nt1 < self.vocab:
                    # nt1 is a micro-action
                    self.micro_execute.append(nt1)
                else:
                    # nt1 is a macro-action
                    self.leftmost.append(nt1)

        return log_prob, rule_prob


model = Policy().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()


def main(max_steps=max_steps, entropy_factor=entropy_factor):
    max_steps = max_steps

    running_reward = 0

    # run inifinitely many episodes
    tic = time.time()
    highest_ep_reward = -1e6

    writer = SummaryWriter()

    for i_episode in range(1, 1001):  # count(1):
        micro_list = []

        # reset environment and episode reward
        state = env.reset()
        state = np.array(state)
        state = torch.from_numpy(state).to(device).unsqueeze(1)
        ep_reward = 0
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, max_steps):  # tqdm(range(1, max_steps)):
            if t > 1:
                # if not at the first timestep, we may not change the state
                model.state_changed = True  #False
            # select action from policy
            log_prob, rule_prob = model(state)

            model.action_probs.append(log_prob)
            model.action_probs_hist.append(rule_prob)

            reward = 0
            if len(model.micro_execute) > 0:
                for _ in range(len(model.micro_execute)):
                    action = model.micro_execute.pop(-1)
                    micro_list.append(action)
                    tmp_state, step_reward, done, _ = env.step(action)
                    reward += step_reward
                    if step_reward:  # > -1:
                        # only change the state when make a legal move
                        model.state_changed = True
                        state = tmp_state
                        state = np.array(state)
                        state = torch.from_numpy(state).to(device).unsqueeze(1)
                    if done:
                        break
            else:
                reward = 0
                done = None

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                # print('episode %d success' % i_episode)
                break
            elif len(model.leftmost) == 0:
                # print('parse tree fully expanded, no more leftmost non-terminal')
                break
        # print(micro_list)

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if ep_reward >= highest_ep_reward:
            highest_ep_reward = ep_reward

        # perform backprop
        R = 0
        policy_losses = []  # list to save actor (policy) loss
        entropy_losses = []
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, R, action_probs_hist in zip(model.action_probs, returns, model.action_probs_hist):
            entropy = torch.distributions.Categorical(probs=action_probs_hist).entropy()
            policy_losses.append(-log_prob * R)
            entropy_losses.append(- entropy_factor * entropy)

        # sum up all the values of policy_losses and value_losses
        episode_policy_loss = torch.stack(policy_losses).sum()
        episode_entropy_loss = torch.stack(entropy_losses).sum()
        episode_loss = episode_policy_loss + episode_entropy_loss

        writer.add_scalar('Episode reward', ep_reward, i_episode)
        writer.add_scalar('Steps per episode', t, i_episode)

        writer.add_scalar('Policy loss', episode_policy_loss, i_episode)
        writer.add_scalar('Entroy loss', episode_entropy_loss, i_episode)

        if done:
            writer.add_scalar('Episode success', 1, i_episode)
        else:
            writer.add_scalar('Episode success', 0, i_episode)

        if len(model.leftmost) == 0:
            # parse tree fully expanded
            writer.add_scalar('Parse tree fully expanded', 1, i_episode)
        else:
            writer.add_scalar('Parse tree fully expanded', 0, i_episode)


        # reset gradients
        optimizer.zero_grad()
        # perform backprop+
        episode_loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.action_probs[:]
        model.leftmost = [0]
        model.micro_execute = []
        model.micro_storage = []
        model.hidden_state = None
        del model.action_probs_hist[:]

        # log results
        if i_episode % args.log_interval == 0:
            toc = time.time()
            print(
                'Episode {} \t Last reward: {:.2f} \t Average reward: {:.2f} \t Highest reward: {:.2f} \t Time taken: {:.2f}s'.format(
                    i_episode, ep_reward, running_reward, highest_ep_reward, toc - tic))
            #             print(kl.mean())
            tic = toc
            highest_ep_reward = -1e6

        # check if we have "solved" the cart pole problem
        if running_reward >= 90:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()