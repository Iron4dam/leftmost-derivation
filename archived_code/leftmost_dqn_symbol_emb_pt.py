# symbol embedding with pre-terminals & pretrain

import argparse
import gym
import gym_hanoi
import numpy as np
import time
from collections import namedtuple
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

max_steps = 500    # max steps the agent can execute per episode
max_episodes = 20000  # max number of episodes
entropy_factor = 0.01  # coefficient multiplying the entropy loss
nt = 6    # number of non-terminals
pt = 6   # number of pre-terminals
use_lstm = False   # use LSTM to encode state sequentially or use a simple neural network encoder to encode state at each timestep
lr = 1e-3
gamma = 0.99   # discount factor
log_interval = 10   # episode interval between training logs
allow_state_unchange = False   # if True, we do not pass in the same state into LSTM/encoder if state unchanged
retain_graph = True if allow_state_unchange else False


# gym-hanoi env settings
num_disks = 3
env_noise = 0.   # transition/action failure probability
state_space_dim = num_disks
action_space_dim = 6   # always 6 actions for 3 poles
env = gym.make("Hanoi-v0")
env.set_env_parameters(num_disks, env_noise, verbose=True)
# env.seed(args.seed)
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=gamma, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=log_interval, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=int, default=lr, metavar='N',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--max-steps', type=int, default=max_steps, metavar='N',
                    help='max steps the agent can execute per episode (default: 300)')
parser.add_argument('--max-episodes', type=int, default=max_episodes, metavar='N',
                    help='max number of episodes (default: 1000)')
parser.add_argument('--entropy-factor', type=float, default=entropy_factor, metavar='N',
                    help='coefficient multiplying the entropy loss (default: 0.05)')
parser.add_argument('--action-space-dim', type=int, default=action_space_dim, metavar='N',
                    help='action space dimension (default: 6)')
parser.add_argument('--state-space-dim', type=int, default=state_space_dim, metavar='N',
                    help='state space dimension (default: 3)')
parser.add_argument('--allow-state-unchange', type=bool, default=allow_state_unchange, metavar='N',
                    help='if True, we do not pass in the same state into LSTM/encoder if state unchanged (default: False)')
parser.add_argument('--retain-graph', type=bool, default=retain_graph, metavar='N',
                    help='same as allow-state-unchange (default: False)')
parser.add_argument('--use-lstm', type=bool, default=use_lstm, metavar='N',
                    help='use LSTM to encode state sequentially or use a simple neural network encoder to encode state at each timestep (default: True)')
parser.add_argument('--nt-states', type=int, default=nt, metavar='N',
                    help='number of non-terminals (default: 6)')
parser.add_argument('--pt-states', type=int, default=pt, metavar='N',
                    help='number of pre-terminals (default: 6)')
args = parser.parse_args()


Transition = namedtuple('Transition',
                        ('nt','state', 'action', 'next_state', 'next_nt', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        

class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class DQN(nn.Module):
    def __init__(self, action_space=args.action_space_dim,
                 state_space=args.state_space_dim*3,   # use one hot encoding for states
                 nt_states=nt,
                 pt_states=pt,
                 lstm_dim=32,   # dimension of lstm hidden states
                 symbol_dim=32,  # dimension of symbol embeddings
                 z_dim=16):   # dimension of the latent state encoding vector (either given by LSTM or a simple NN encoder)

        super(DQN, self).__init__()

        # dimensions
        self.action_space = action_space
        self.state_space = state_space
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.symbol_dim = symbol_dim
        self.z_dim = z_dim

        # embeddings
        self.root_rule_num = nt_states * nt_states
        self.nt_rule_num = nt_states * (nt_states + pt_states) * (nt_states + pt_states)
        self.nt_nt_rule_num = (nt_states + pt_states) * (nt_states + pt_states)
        self.pt_rule_num = action_space

        self.root_emb = nn.Parameter(torch.randn(1, symbol_dim))
        self.nt_emb = nn.Parameter(torch.randn(nt_states, symbol_dim))
        self.pt_emb = nn.Parameter(torch.randn(pt_states, symbol_dim))

        self.register_parameter('root_emb', self.root_emb)
        self.register_parameter('nt_emb', self.nt_emb)


        # encoding state directly to latent state vector z
        self.state_emb_z = nn.Linear(self.state_space, self.z_dim)

        # value functions
        self.root_values = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      nn.Linear(self.z_dim, self.root_rule_num))
        self.nt_values = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      nn.Linear(self.z_dim, self.nt_nt_rule_num))
        self.pt_values = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      nn.Linear(self.z_dim, self.pt_rule_num))

    def forward(self, nt, state, use_mean=True):
        # change state to one hot vector
        state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
        state_emb = state_one_hot.scatter(1, state, 1).view(1, -1)    # 1 x (3 x state_space)

        self.z = F.relu(self.state_emb_z(state_emb))
        z = self.z

        # pop the leftmost non-terminal
        assert nt < self.action_space + self.nt_states + self.pt_states
        assert nt == 0 or nt >= self.action_space   # assert nt is either the root or nt/pt, not a terminalaz

        # calculate rule probs
        if nt == 0:
            # the leftmost node is a root node
            root_emb = self.root_emb   # 1 x symbol_dim
            z = z.expand(1, self.z_dim)
            root_emb = torch.cat([root_emb, z], 1)

            # calculate action/rule-value
            values = self.root_values(root_emb)

        else:
            # the leftmost node is a non-terminal (nt or pt)
            nt = nt - self.action_space

            if nt < self.nt_states:   # nt is a nt
                # z = z.expand(self.nt_nt_rule_num, self.z_dim)
                nt_emb = self.nt_emb   # nt_nt_rule_num x symbol_dim
                # nt_rule_emb = nt_rule_emb.expand(self.nt_rule_num, self.symbol_dim).view(
                    # self.nt_states,(self.action_space + self.nt_states)*(self.action_space + self.nt_states),self.symbol_dim)
                nt_emb = nt_emb[nt,:].unsqueeze(0)  #  nt_nt_rules_num x symbol_dim
                # print(nt_emb.shape,z.shape)
                nt_emb = torch.cat([nt_emb, z], 1)

                # calculate action/rule-value
                values = self.nt_values(nt_emb)

            else:   # nt is a pt
                pt = nt - self.nt_states
                pt_emb = self.pt_emb
                pt_emb = pt_emb[pt,:].unsqueeze(0)
                pt_emb = torch.cat([pt_emb, z], 1)

                # calculate action/rule-value
                values = self.pt_values(pt_emb)

        # return log_prob for the chosen rule (for policy loss calculation), and return rule prob for all rules that can be chosen at this step (for entropy calculation)
        return values




def main():
    Transition = namedtuple('Transition',
                        ('nt','state', 'rule', 'next_state', 'next_nt', 'reward'))

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    memory = ReplayMemory(1000 )

    eps = np.finfo(np.float32).eps.item()
    writer = SummaryWriter()  #'/home/adamxinyuyang/Documents/leftmost-derivation/runs_dqn', comment='DQN')

    tic = time.time()
    running_reward = 0
    highest_ep_reward = -1e6   # record the highest episode reward in each log-interval

    success_count = 0
    for i_episode in range(1, args.max_episodes + 1):  # count(1):
        micro_list = []   # store the list of action sequence for each episode, in case we need to check

        leftmost = [[0]]
        micro_execute = []
        nt = 0
        
        # reset environment and episode reward
        state = env.reset()
        state = np.array(state)
        state = torch.from_numpy(state).to(device).unsqueeze(1)
        ep_reward = 0

        for t in range(1, args.max_steps + 1):    # tqdm(range(1, max_steps)):
            # pop the leftmost non-terminal

            # select eps-greedy action
            eps_sample = random.random()
            eps_threshold = 0.1
            if eps_sample > eps_threshold:
                with torch.no_grad():
                    values = policy_net(nt, state)
                    rule = values.max(1)[1]
            else:
                if nt == 0:
                    action_space = args.nt_states**2
                else:
                    nt_tmp = nt - args.action_space_dim
                    if nt_tmp < args.nt_states:
                        action_space = (args.nt_states + args.pt_states)**2
                    else:
                        action_space = args.action_space_dim
                    
                rule = torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)
            if nt == 0:
                # del leftmost[-1]
                # del micro_storage[-1]
                leftmost.append([0])

                nt1 = int((rule//args.nt_states) + args.action_space_dim)
                nt2 = int((rule%args.nt_states) + args.action_space_dim)

                leftmost.append([nt1,nt2])
            
            else:
                nt_tmp = nt - args.action_space_dim

                if nt_tmp < args.nt_states:   # nt is a nt
                    nt1 = int(rule//(args.pt_states + args.nt_states) + args.action_space_dim)
                    nt2 = int(rule%(args.pt_states + args.nt_states) + args.action_space_dim)

                    leftmost.append([nt1,nt2])

                else:  # nt is a pt
                    micro_execute.append(int(rule))

            reward = 0
            if len(micro_execute) > 0:
                action = micro_execute.pop(0)
                micro_list.append(action)
                next_state, step_reward, done, _ = env.step(action)
                next_state = np.array(next_state)
                next_state = torch.from_numpy(next_state).to(device).unsqueeze(1)

                if step_reward == -1:
                    step_reward = -0.01
                if step_reward == 0:
                    step_reward = 0.01
                if step_reward == 100:
                    step_reward = 5
                reward += step_reward
                if done:
                    break
            else:
                reward = 0
                next_state = state
                done = None
            
            ep_reward += reward
            
            if done:
                # if len(model.leftmost) == 0:
                    # model.rewards[-1] += 50
                    # ep_reward += 50
                break
            elif len(leftmost) == 0:   # if no more unexpanded leftmost non-terminals (the parse tree has fully expanded), but not success
                # model.rewards[-1] -= 50
                # ep_reward -= 50
                break
            next_nt = leftmost[-1].pop(0)
            if len(leftmost[-1]) == 0:
                del leftmost[-1]

            # Store the transition in memory
            memory.push(nt, state, rule, next_state, next_nt, reward)

            # values = policy_net(nt, state).squeeze()
            # value = values[rule]
            # next_values = target_net(next_nt, next_state)
            # next_value = next_values.max(1)[0].detach().squeeze()
            # expected_value = next_value * args.gamma + reward

            transitions = memory.sample(1)   # bs = 1
            batch = Transition(*zip(*transitions))

            values = policy_net(batch.nt[0], batch.state[0]).squeeze()
            value = values[batch.rule[0]]
            next_values = target_net(batch.next_nt[0], batch.next_state[0])
            next_value = next_values.max(1)[0].detach().squeeze()
            expected_value = next_value * args.gamma + batch.reward[0]

            state = next_state
            nt = next_nt

            loss = F.smooth_l1_loss(value.squeeze(), expected_value.squeeze())

            optimizer.zero_grad()
            loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()


        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if running_reward >= 5:
            success_count += 1
        else:
            success_count = 0

        if success_count > 200:
            # agent solved the game
            torch.save(model.state_dict(), '/home/adamxinyuyang/Documents/leftmost-derivation/saved_models/%.3f_entropy_%d_nt_%d_pt_%s_lstm.pt' %(args.entropy_factor, args.nt_states, args.pt_states, args.use_lstm))
            print('Agent solved the game!')
            break

        if ep_reward >= highest_ep_reward:
            highest_ep_reward = ep_reward

        if ep_reward == 0:
            print(micro_list)


        # write to tensorboard
        writer.add_scalar('Episode reward', ep_reward, i_episode)
        writer.add_scalar('Steps per episode', t, i_episode)

        if done:
            writer.add_scalar('Episode success', 1, i_episode)
        else:
            writer.add_scalar('Episode success', 0, i_episode)

        if len(leftmost) == 0:
            # parse tree fully expanded
            writer.add_scalar('Parse tree fully expanded', 1, i_episode)
        else:
            writer.add_scalar('Parse tree fully expanded', 0, i_episode)

        # reset rewards and action buffer
        leftmost = [[0]]
        micro_execute = []
        micro_storage = [[]]

        # log results
        if i_episode % args.log_interval == 0:
            toc = time.time()
            print(
                'Episode {} \t Last reward: {:.2f} \t Average reward: {:.2f} \t Highest reward: {:.2f} \t Time taken: {:.2f}s'.format(
                    i_episode, ep_reward, running_reward, highest_ep_reward, toc - tic))
            tic = toc
            highest_ep_reward = -1e6



if __name__ == '__main__':
    main()