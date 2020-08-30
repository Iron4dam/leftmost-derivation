# scales rule probabilities towards preterminals proportional to node count

import argparse
import gym
import gym_hanoi
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

max_steps = 1000    # max steps the agent can execute per episode
max_episodes = 20000  # max number of episodes
nt = 6   # number of non-terminals
pt = 6   # number of pre-terminals

rule_reward = -0.00
positive_reward = 0.01
negative_reward = -0.01
node_scale = 0.

entropy_factor = 0.1  # coefficient multiplying the entropy loss
decay = True
decay_scale = 0.8

lr = 1e-3
gamma = 0.99   # discount factor

use_lstm = False   # use LSTM to encode state sequentially or use a simple neural network encoder to encode state at each timestep
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
env.seed(0)
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")




parser = argparse.ArgumentParser(description='PyTorch actor-critic example')

parser.add_argument('--decay', type=bool, default=decay)
parser.add_argument('--decay-scale', type=float, default=decay_scale)
parser.add_argument('--rule-reward', type=float, default=rule_reward)
parser.add_argument('--positive-reward', type=float, default=positive_reward)
parser.add_argument('--negative-reward', type=float, default=negative_reward)
parser.add_argument('--scale', type=float, default=node_scale)

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


class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class Critic(nn.Module):
    def __init__(self, state_space=args.state_space_dim*3):
        super(Critic, self).__init__()
        # self.root_value = nn.Sequential(nn.Linear(state_space, 32),
        #                                     ResidualLayer(32, 32),
        #                                     ResidualLayer(32, 32),
        #                                     nn.Linear(32, 1))
        # self.nt_value = nn.Sequential(nn.Linear(state_space, 32),
        #                                     ResidualLayer(32, 32),
        #                                     ResidualLayer(32, 32),
        #                                     nn.Linear(32, 1))
        # self.pt_value = nn.Sequential(nn.Linear(state_space, 32),
        #                                     ResidualLayer(32, 32),
        #                                     ResidualLayer(32, 32),
        #                                     nn.Linear(32, 1))
        self.value_network = nn.Sequential(nn.Linear(state_space, 32),
                                            ResidualLayer(32, 32),
                                            ResidualLayer(32, 32),
                                            nn.Linear(32, 1))
        self.state_space = state_space
    
    def forward(self, state, node_type):
        # change state to one hot vector
        state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
        state_emb = state_one_hot.scatter(1, state, 1).view(1, -1)    # 1 x (3 x state_space)
        
        value = self.value_network(state_emb)

        # if node_type == 'S':
        #     value = self.root_value(state_emb)
        # elif node_type == 'NT':
        #     value = self.nt_value(state_emb)
        # elif node_type == 'PT':
        #     value = self.pt_value(state_emb)
        # else: 
        #     raise NotImplementedError
        return value


class Policy(nn.Module):
    def __init__(self, action_space=args.action_space_dim,
                 state_space=args.state_space_dim*3,   # use one hot encoding for states
                 nt_states=nt,
                 pt_states=pt,
                 lstm_dim=32,   # dimension of lstm hidden states
                 symbol_dim=32,  # dimension of symbol embeddings
                 z_dim=16,   # dimension of the latent state encoding vector (either given by LSTM or a simple NN encoder)
                 use_lstm = args.use_lstm,
                 allow_state_unchange = args.allow_state_unchange):
        super(Policy, self).__init__()

        # dimensions
        self.action_space = action_space
        self.state_space = state_space
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.symbol_dim = symbol_dim
        self.z_dim = z_dim
        self.use_lstm = use_lstm
        self.allow_state_unchange = allow_state_unchange

        # action probs, micro/macro action, reward, hidden state buffer
        # self.action_probs = []
        # self.action_probs_hist = []
        # self.values = []
        # self.leftmost = [[0]]   # leftmost unexpanded non-terminal, append a list of non-terminal(s) for each expansion
        # self.micro_execute = []  # micro action(s) to execute at this time step
        # self.rewards = []
        self.hidden_state = None
        self.state_changed = True   # stores whether state changed in the last timestep

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

        # encoders, NNs
        if use_lstm:
            # encoding state through a LSTM
            self.enc_lstm = nn.LSTM(self.state_space, lstm_dim, bidirectional=False, num_layers=1, batch_first=True)
            self.lstm_z = nn.Linear(lstm_dim, z_dim * 2)
        else:
            # encoding state directly to latent state vector z
            self.state_emb_z = nn.Linear(self.state_space, self.z_dim)
        # output a single rule score for each rule (root rule: S->NT*NT, nt rule: NT->(NT*T)*(NT*T)
        self.root_rule_mlp = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      nn.Linear(symbol_dim, self.root_rule_num))
        # self.nt_rule_mlp = nn.Linear(symbol_dim + self.z_dim, self.nt_nt_rule_num)
        self.nt_rule_mlp = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      nn.Linear(symbol_dim, self.nt_nt_rule_num))
        self.pt_rule_mlp = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      nn.Linear(symbol_dim, action_space))

    def forward(self, nt, state, node_count, scale, use_mean=True):
        # change state to one hot vector
        state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
        state_emb = state_one_hot.scatter(1, state, 1).view(1, -1)    # 1 x (3 x state_space)

        if self.use_lstm:
            if self.state_changed or (not self.allow_state_unchange):
                # state has changed, or we do not allow to use the last state in the previous step, pass new state into LSTM
                h, hidden_state = self.enc_lstm(state_emb.unsqueeze(0), self.hidden_state)
                self.hidden_state = hidden_state   # hidden state buffer
                params = self.lstm_z(h.max(1)[0])
                mean = params[:, :self.z_dim]
                logvar = params[:, self.z_dim:]

                if use_mean:
                    z = mean
                else:
                    eps = mean.new(1, self.z_dim).normal(0, 1)
                    z = (0.5 * logvar).exp() * eps + mean
                self.z = z
            elif (not self.state_changed) and allow_state_unchange:
                # state hasn't changed, and we allow to use the last state in previous step, use the same latent state vector z as before
                z = self.z
        else:
            # use NN to encode z
            if self.state_changed or (not self.allow_state_unchange):
                # state has changed, or we do not allow to use the last state in the previous step, pass new state into LSTM
                self.z = F.relu(self.state_emb_z(state_emb))
                z = self.z
            elif (not self.state_changed) and allow_state_unchange:
                # state hasn't changed, and we allow to use the last state in previous step, use the same latent state vector z as before
                z = self.z


        # calculate rule probs
        if nt == 'S':

            # self.leftmost.append([0])

            # the leftmost node is a root node
            root_emb = self.root_emb   # 1 x symbol_dim
            z = z.expand(1, self.z_dim)
            root_emb = torch.cat([root_emb, z], 1)
            root_rule_score = self.root_rule_mlp(root_emb).squeeze()
            rule_prob = F.softmax(root_rule_score)  # root_rule_num x 1

            # select expansion rule
            m = Categorical(rule_prob)
            rule = m.sample()
            log_prob = m.log_prob(rule)

            nt1 = int((rule//self.nt_states) + self.action_space)
            nt2 = int((rule%self.nt_states) + self.action_space)

            children_list = [nt1,nt2]

            left_node = nt1
            right_node = nt2

        else:
            assert nt >= self.action_space
            assert nt < self.action_space  + self.nt_states + self.pt_states
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
                nt_rule_score = self.nt_rule_mlp(nt_emb).squeeze()

                # scale rule scores for PT rules proportional to node cound
                nt_rule_score[self.nt_states**2-1:] += node_count*scale

                rule_prob = F.softmax(nt_rule_score)

                # select expansion rule
                m = Categorical(rule_prob)
                rule = m.sample()
                log_prob = m.log_prob(rule)

                nt1 = int(rule//(self.pt_states + self.nt_states) + self.action_space)
                nt2 = int(rule%(self.pt_states + self.nt_states) + self.action_space)

                left_node = nt1
                right_node = nt2

            else:   # nt is a pt
                pt = nt - self.nt_states
                pt_emb = self.pt_emb
                pt_emb = pt_emb[pt,:].unsqueeze(0)

                pt_emb = torch.cat([pt_emb, z], 1)
                pt_rule_score = self.pt_rule_mlp(pt_emb).squeeze()
                rule_prob = F.softmax(pt_rule_score)

                # select expansion rule
                m = Categorical(rule_prob)
                rule = m.sample()
                log_prob = m.log_prob(rule)

                a = int(rule)

                left_node = a
                right_node = None

        # return log_prob for the chosen rule (for policy loss calculation), and return rule prob for all rules that can be chosen at this step (for entropy calculation)
        return log_prob, rule_prob, left_node, right_node


def calculate_return(reward_list, gamma):
    reward_list = [item for sublist in reward_list for item in sublist]

    discounted_return = 0
    for R in reversed(reward_list):
        discounted_return = R + gamma*discounted_return
    return discounted_return


class Tree:
    def __init__(self, parent_tree, 
                left_node=None, 
                right_node=None, 
                step=None,
                action_space=args.action_space_dim,
                nt_states=args.nt_states,
                pt_states=args.pt_states):
        self.parent_tree = parent_tree
        self.left_node = left_node
        self.right_node = right_node
        self.step = step   # step in episode, to calculate loss later

        self.action_space = action_space
        self.nt_states = nt_states
        self.pt_states = pt_states

        self.left_node_type = self.node_type(left_node)
        self.right_node_type = self.node_type(right_node)
        self.left_tree = None
        self.right_tree = None

        # just for initialisation, change later with reevaluation()
        if self.left_node_type == 'T':  # left node is a terminal
            self.expanded = True
        else:
            self.expanded = False

        if self.parent_tree == 'S':
            self.node_count = 0
        else:
            self.node_count = self.parent_tree.node_count + 1

    def node_type(self,node):
        if node is not None:
            if node < self.action_space:
                return 'T'
            elif node < self.action_space + self.nt_states:
                return 'NT'
            else:
                return 'PT'
        else:
            return None

    def reevaluate(self):
        if isinstance(self.left_tree, Tree) and isinstance(self.right_tree, Tree):
            # parent=NT/S, and both children are Trees
            if (self.left_tree.expanded == True) and (self.right_tree.expanded == True):
                return True
            else:
                return False


def calculate_advantage(value_list, value_new, reward_list, back_prop_idx, gamma):
    tmp_back_prop_idx = back_prop_idx.copy()
    old_value_list = [-value_list[i] for i in tmp_back_prop_idx]
    old_value_tensor = torch.stack(old_value_list)
    return_list = []
    total_return = value_new
    for i,reward in enumerate(reversed(reward_list)):
        total_return = reward + gamma*total_return
        idx = len(reward_list) - 1 - i
        if tmp_back_prop_idx[0] == idx:
            return_list.append(total_return)
            tmp_back_prop_idx.pop(0)
        if len(tmp_back_prop_idx) == 0:
            break
    return_tensor = torch.stack(return_list)
    advantage_tensor = return_tensor - old_value_tensor
    return advantage_tensor


def linear_anneal(counter, start, final, in_steps):
    out = max(final, start + counter * (final-start) / in_steps)
    return out


def main():
    model = Policy().to(device)
    critic = Critic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    writer = SummaryWriter(log_dir='/home/xy2419/leftmost-derivation/runs/span_all_value_%.3f_%.1f_entropy_%d_nt_%d_pt_%s_decay_%.3f_%.3f_%.3f_reward_%d_disks_%d_%.3f_scale/' 
        %(args.entropy_factor, args.decay_scale, args.nt_states, args.pt_states, args.decay, args.rule_reward, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps, args.scale))

    tic = time.time()
    running_reward = 0
    highest_ep_reward = -1e6   # record the highest episode reward in each log-interval

    success_count = 0
    episode_entropy_loss = []
    episode_actor_loss = []
    episode_critic_loss = []
    entropy_factor = args.entropy_factor
    min_step = 1e5
    for i_episode in range(1, args.max_episodes+1):  # count(1):

        if args.decay:
            entropy_factor = linear_anneal(counter=i_episode, start=args.entropy_factor, final=0.01, in_steps=1e4)

        micro_list = []   # store the list of action sequence for each episode, in case we need to check

        # reset environment and episode reward
        state = env.reset()
        state = np.array(state)
        state = torch.from_numpy(state).to(device).unsqueeze(1)

        parent_node = 'S'
        parent_node_type = 'S'
        parent_tree = None
        ep_reward = 0
        num_tree = 1
        max_node_count = 0

        value_list = []
        reward_list = []
        action_logprob_list = []
        action_dist_list = []

        for t in range(args.max_steps):    # tqdm(range(1, max_steps)):
            old_value = critic(state, parent_node_type)
            value_list.append(old_value)

            # select action from policy
            if parent_tree is None:
                node_count = 0
            else: 
                node_count = parent_tree.node_count
            
            if node_count > max_node_count:
                max_node_count = node_count

            log_prob, rule_prob, left_node, right_node = model(parent_node, state, scale=args.scale, node_count=node_count)
            action_logprob_list.append(log_prob)
            action_dist_list.append(rule_prob)

            if parent_node ==  'S':
                child_tree = Tree(parent_node, left_node, right_node, step=t)
            else:
                child_tree = Tree(parent_tree, left_node, right_node, step=t)
                if parent_tree.left_tree is None:
                    parent_tree.left_tree = child_tree
                else:
                    parent_tree.right_tree = child_tree
            

            reward = args.rule_reward
            done = None
            if right_node == None:
                # arrived at a left_node terminal
                assert child_tree.expanded
                assert child_tree.left_node_type == 'T'
                action = left_node
                micro_list.append(action)
                tmp_state, step_reward, done, _ = env.step(action)
                if step_reward == -1:
                    step_reward = args.negative_reward
                if step_reward == 0:
                    step_reward = args.positive_reward
                if step_reward == 100:
                    step_reward = 5 
                reward += step_reward
                reward_list.append(reward)

                state = tmp_state
                state = np.array(state)
                state = torch.from_numpy(state).to(device).unsqueeze(1)
                # value_list.append(value_new)
                
                back_prop_idx = [t]
                parent_tree.expanded = parent_tree.reevaluate()
                while parent_tree.expanded == True:
                    back_prop_idx.append(parent_tree.step)
                    if parent_tree.parent_tree == 'S':
                        # arrived at the first branch on the tree, append backprop list and stop here
                        break
                    else:
                        parent_tree = parent_tree.parent_tree
                        parent_tree.expanded = parent_tree.reevaluate()
                
                if parent_tree.parent_tree == 'S':
                    if parent_tree.expanded:
                    # if the whole tree expanded, start a new tree
                        parent_node = 'S'
                        parent_node_type = 'S'
                        parent_tree = None
                        num_tree += 1
                    else:
                        parent_node = parent_tree.right_node
                        parent_node_type = parent_tree.right_node_type
                        parent_tree = parent_tree
                else:
                    parent_node = parent_tree.right_node
                    parent_node_type = parent_tree.right_node_type
                    parent_tree = parent_tree
                
                value_new = critic(state, parent_node_type)

                advantage_tensor = calculate_advantage(value_list, value_new, reward_list, back_prop_idx, gamma=args.gamma)
                advantage = torch.mean(advantage_tensor**2)
                critic_loss = advantage
                episode_critic_loss.append(critic_loss.item())
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                dist_list = [action_dist_list[i] for i in back_prop_idx]
                logprob_list = [action_logprob_list[i] for i in back_prop_idx]

                entropy_list = [torch.distributions.Categorical(probs=dist).entropy() for dist in dist_list]
                entropy_tensor = torch.stack(entropy_list)
                entropy_loss = - entropy_factor * entropy_tensor.mean()
                episode_entropy_loss.append(entropy_loss.item())

                logprob_tensor = torch.stack(logprob_list)
                actor_loss = torch.mean(-logprob_tensor * advantage.detach())
                episode_actor_loss.append(actor_loss.item())
                optimizer.zero_grad()
                (actor_loss + entropy_loss).backward()
                optimizer.step()

                # if parent_tree.parent_tree == 'S':
                #     if parent_tree.expanded:
                #     # if the whole tree expanded, start a new tree
                #         parent_node = 'S'
                #         parent_node_type = 'S'
                #         parent_tree = None
                #         num_tree += 1
                #     else:
                #         parent_node = parent_tree.right_node
                #         parent_node_type = parent_tree.right_node_type
                #         parent_tree = parent_tree
                # else:
                #     parent_node = parent_tree.right_node
                #     parent_node_type = parent_tree.right_node_type
                #     parent_tree = parent_tree

            else:
                # no terminal to execute, no backprop, continue next step
                # value_new = critic(state, child_tree.left_node_type)
                parent_node = left_node
                parent_node_type = child_tree.left_node_type
                parent_tree = child_tree
                state = state
                reward_list.append(reward)

            ep_reward += reward

            if done:
                break



        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward


        if ep_reward >= highest_ep_reward:
            highest_ep_reward = ep_reward

      
        # write to tensorboard
        writer.add_scalar('4 Episode reward', ep_reward, i_episode)
        writer.add_scalar('6 Steps per episode', t, i_episode)

        writer.add_scalar('1 Actor loss', np.mean(episode_actor_loss), i_episode)
        writer.add_scalar('2 Entropy loss', np.mean(episode_entropy_loss), i_episode)
        writer.add_scalar('3 Critic loss', np.mean(episode_critic_loss), i_episode)

        if done:
            writer.add_scalar('5 Episode success', 1, i_episode)
        else:
            writer.add_scalar('5 Episode success', 0, i_episode)

        writer.add_scalar('7 Number of trees', num_tree, i_episode)
        writer.add_scalar('8 Length of terminal sequence', len(micro_list), i_episode)
        writer.add_scalar('9 Max node count', max_node_count, i_episode)


        # log results
        if i_episode % args.log_interval == 0:
            toc = time.time()
            print(
                'Episode {} \t Last reward: {:.2f} \t Average reward: {:.2f} \t Highest reward: {:.2f} \t Time taken: {:.2f}s'.format(
                    i_episode, ep_reward, running_reward, highest_ep_reward, toc - tic))
            #             print(kl.mean())
            tic = toc
            highest_ep_reward = -1e6
        

        if done:
            success_count += 1
        else:
            success_count = 0

        if i_episode > 4:
            # agent solved the game
            if t < min_step:
                torch.save(model.state_dict(), '/home/xy2419/leftmost-derivation/saved_models/span_all_value_%.3f_entropy_%d_nt_%d_pt_%s_decay_%.3f_%.3f_%.3f_reward_%d_disks_%d_steps.pt' 
                %(args.entropy_factor, args.nt_states, args.pt_states, args.decay, args.rule_reward, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps))
                print('step taken %d, model saved' %t)
                min_step = t


if __name__ == '__main__':
    main()