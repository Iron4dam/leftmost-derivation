# symbol embedding with pre-terminals & pretrain

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
use_lstm = False   # use LSTM to encode state sequentially or use a simple neural network encoder to encode state at each timestep
lr = 1e-3
gamma = 0.99   # discount factor
log_interval = 10   # episode interval between training logs


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

entropy_factor = 0.1  # coefficient multiplying the entropy loss
decay = True
value_one = False

rule_reward = -0.00
positive_reward = 0.01
negative_reward = -0.01

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')

parser.add_argument('--decay', type=bool, default=decay)
parser.add_argument('--rule-reward', type=float, default=rule_reward)
parser.add_argument('--positive-reward', type=float, default=positive_reward)
parser.add_argument('--negative-reward', type=float, default=negative_reward)
parser.add_argument('--value-one', type=bool, default=value_one)

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

class Policy(nn.Module):
    def __init__(self, action_space=args.action_space_dim,
                 state_space=args.state_space_dim*3,   # use one hot encoding for states
                 nt_states=nt,
                 pt_states=pt,
                 lstm_dim=32,   # dimension of lstm hidden states
                 symbol_dim=32,  # dimension of symbol embeddings
                 z_dim=16,   # dimension of the latent state encoding vector (either given by LSTM or a simple NN encoder)
                 use_lstm = args.use_lstm,
                 value_one = args.value_one):
        super(Policy, self).__init__()

        # dimensions
        self.action_space = action_space
        self.state_space = state_space
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.symbol_dim = symbol_dim
        self.z_dim = z_dim
        self.use_lstm = use_lstm
        self.num_trees = 0
        self.value_one = value_one

        self.action_hist = []
        self.graph = ''
        self.graph_count = 1
        self.first_S = None

        # action probs, micro/macro action, reward, hidden state buffer
        self.action_probs = []
        self.action_probs_hist = []
        self.values = []
        self.leftmost = [[(0,1)]]   # leftmost unexpanded non-terminal, append a list of non-terminal(s) for each expansion
        self.micro_execute = []  # micro action(s) to execute at this time step
        self.rewards = []
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

        # policy networks
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

        # value networks
        if value_one:
            self.values_all = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      ResidualLayer(self.z_dim, self.z_dim),
                                      nn.Linear(self.z_dim, 1))

        else:
            self.root_values = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                        ResidualLayer(self.z_dim, self.z_dim),
                                        ResidualLayer(self.z_dim, self.z_dim),
                                        nn.Linear(self.z_dim, 1))
            self.nt_values = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                        ResidualLayer(self.z_dim, self.z_dim),
                                        ResidualLayer(self.z_dim, self.z_dim),
                                        nn.Linear(self.z_dim, 1))
            self.pt_values = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, self.z_dim),
                                        ResidualLayer(self.z_dim, self.z_dim),
                                        ResidualLayer(self.z_dim, self.z_dim),
                                        nn.Linear(self.z_dim, 1))

    def forward(self, state, use_mean=True):
        assert len(self.micro_execute) == 0   # no unexecuted micro-actions

        # change state to one hot vector
        state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
        state_emb = state_one_hot.scatter(1, state, 1).view(1, -1)    # 1 x (3 x state_space)

        if self.use_lstm:
            if self.state_changed:
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
            elif (not self.state_changed):
                # state hasn't changed, and we allow to use the last state in previous step, use the same latent state vector z as before
                z = self.z
        else:
            # use NN to encode z
            if self.state_changed:
                # state has changed, or we do not allow to use the last state in the previous step, pass new state into LSTM
                self.z = F.relu(self.state_emb_z(state_emb))
                z = self.z
            elif (not self.state_changed):
                # state hasn't changed, and we allow to use the last state in previous step, use the same latent state vector z as before
                z = self.z


        # pop the leftmost non-terminal
        leftmost_pop = self.leftmost[-1].pop(0)
        nt = leftmost_pop[0]
        nt_count = leftmost_pop[1]
        if len(self.leftmost[-1]) == 0:
            del self.leftmost[-1]
        assert nt == 0 or nt >= self.action_space   # assert nt is either the root or nt/pt, not a terminal
        # assert len(self.leftmost) == len(self.micro_storage)   # append a list of micros/macros at each hierarchy even if there're none, so they should be of same length

        # calculate rule probs
        if nt == 0:
            self.num_trees += 1
            self.leftmost.append([(0,self.graph_count+3)])

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

            nt1_string = 'NT' + str(nt1-self.action_space)
            nt2_string = 'NT' + str(nt2-self.action_space)
            rule_string = '0' + '->' + nt1_string + nt2_string
            prob_string = '%.3f' %rule_prob[rule]   #torch.exp(log_prob)   #rule_prob[rule]
            self.action_hist.append((rule_string,prob_string))

            n0 = 'n' + str(nt_count)
            n1 = 'n' + str(self.graph_count+1)
            n2 = 'n' + str(self.graph_count+2)
            # self.graph += '\n' + n0+'[label="S"]' + '\n'+n0+'--'+n1 + '\n' + n0+'--'+n2 + '\n' \
            #     + n1+'[label="%s"]' + '\n' + n2+'[label="%s"]' + '\n' %(nt1_string, nt2_string)
            if len(self.graph) > 0:
                self.graph += '\n' + self.first_S + '--' + n0
            self.first_S = n0

            self.graph += '\n' + n0+'[label="S"]' + '\n'+n0+'--'+n1+f'[label={prob_string}]' \
                + '\n' + n0+'--'+n2 + '\n' \
                + n1+f'[label="{nt1_string}"]' + '\n' + n2+f'[label="{nt2_string}"]' + '\n' 

            self.leftmost.append([(nt1,self.graph_count+1),(nt2,self.graph_count+2)])
            self.graph_count += 3

            # calculate action/rule-value
            if self.value_one:
                values = self.values_all(root_emb)
                value = values[:]
            else:
                values = self.root_values(root_emb)
                value = values[:]


        else:
            assert nt >= self.action_space
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
                rule_prob = F.softmax(nt_rule_score)

                # select expansion rule
                m = Categorical(rule_prob)
                rule = m.sample()
                log_prob = m.log_prob(rule)

                nt1 = int(rule//(self.pt_states + self.nt_states) + self.action_space)
                nt2 = int(rule%(self.pt_states + self.nt_states) + self.action_space)

                nt1_tmp = nt1 - self.action_space
                nt2_tmp = nt2 - self.action_space
                if nt1_tmp < self.nt_states:  # nt1_tmp is a NT
                    nt1_tmp = 'NT' + str(nt1_tmp)
                else:  # nt1_tmp is a PT
                    nt1_tmp = nt1_tmp - self.nt_states
                    nt1_tmp = 'PT' + str(nt1_tmp)
                if nt2_tmp < self.nt_states:  # nt2_tmp is a NT
                    nt2_tmp = 'NT' + str(nt2_tmp)
                else:  # nt2_tmp is a PT
                    nt2_tmp = nt2_tmp - self.nt_states
                    nt2_tmp = 'PT' + str(nt2_tmp)

                rule_string = 'NT' + str(nt) + '->' + nt1_tmp + nt2_tmp
                prob_string = '%.3f' %rule_prob[rule]   #torch.exp(log_prob)   #rule_prob[rule]
                self.action_hist.append((rule_string,prob_string))

                n0 = 'n' + str(nt_count)
                n1 = 'n' + str(self.graph_count+1)
                n2 = 'n' + str(self.graph_count+2)
                # self.graph += '\n' + n0+'[label="NT%d"]' + '\n'+n0+'--'+n1 + '\n' + n0+'--'+n2 + '\n' \
                #     + n1+'[label="%s"]' + '\n' + n2+'[label="%s"]' + '\n' %(nt, nt1_tmp, nt2_tmp)
                self.graph += '\n' + n0+f'[label="NT{nt}"]' + '\n'+n0+'--'+n1+f'[label={prob_string}]' \
                    + '\n' + n0+'--'+n2 + '\n' \
                    + n1+f'[label="{nt1_tmp}"]' + '\n' + n2+f'[label="{nt2_tmp}"]' + '\n' 

                self.leftmost.append([(nt1,self.graph_count+1), (nt2,self.graph_count+2)])
                self.graph_count += 2

                # calculate action/rule-value
                if self.value_one:
                    values = self.values_all(nt_emb)
                    value = values[:]
                else:
                    values = self.nt_values(nt_emb)
                    value = values[:]

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

                # calculate action/rule-value
                if self.value_one:
                    values = self.values_all(pt_emb)
                    value = values[:]
                else:
                    values = self.nt_values(pt_emb)
                    value = values[:]

                a = int(rule)
                self.micro_execute.append(a)

                rule_string = 'PT' + str(pt) + '->' + 'T' + str(a)
                prob_string = '%.3f' %rule_prob[rule]   #torch.exp(log_prob)   #rule_prob[rule]
                self.action_hist.append((rule_string,prob_string))

                action_string = ['a','b','c','d','e','f'][a]

                n0 = 'n' + str(nt_count)
                n1 = 'n' + str(self.graph_count+1)
                # self.graph += '\n' + n0+'[label="PT%d"]' + '\n'+n0+'--'+n1 + '\n' \
                #     + n1+'[label="T%d"]' + '\n' %(pt, a)
                self.graph += '\n' + n0+f'[label="PT{pt}"]' + '\n'+n0+'--'+n1+f'[label={prob_string}]' + '\n' \
                    + n1+f'[label="{action_string}"]' + '\n'
                self.graph_count += 1

        # return log_prob for the chosen rule (for policy loss calculation), and return rule prob for all rules that can be chosen at this step (for entropy calculation)
        return log_prob, rule_prob, value


def linear_anneal(counter, start, final, in_steps):
    out = max(final, start + counter * (final-start) / in_steps)
    return out


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model = Policy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    writer = SummaryWriter(log_dir='/home/xy2419/leftmost-derivation/runs/NN_%s_value_%.3f_entropy_%d_nt_%d_pt_%s_decay_%.3f_%.3f_%.3f_reward_%d_disks_%d_steps/' 
        %(args.value_one, args.entropy_factor, args.nt_states, args.pt_states, args.decay, args.rule_reward, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps))


    tic = time.time()
    running_reward = 0
    highest_ep_reward = -1e6   # record the highest episode reward in each log-interval

    success_count = 0
    entropy_factor = args.entropy_factor
    min_step = 1e5
    for i_episode in range(1, args.max_episodes + 1):  # count(1):
        if args.decay:
            entropy_factor = linear_anneal(counter=i_episode, start=args.entropy_factor, final=0.01, in_steps=1e4)

        micro_list = []   # store the list of action sequence for each episode, in case we need to check
        micro_str = ''

        # reset environment and episode reward
        state = env.reset()
        state = np.array(state)
        state = torch.from_numpy(state).to(device).unsqueeze(1)
        ep_reward = 0

        for t in range(1, args.max_steps + 1):    # tqdm(range(1, max_steps)):
            # select action from policy
            log_prob, rule_prob, value = model(state)
            model.action_probs.append(log_prob)
            model.action_probs_hist.append(rule_prob)
            model.values.append(value)

            reward = 0
            if len(model.micro_execute) > 0:
                # for _ in range(len(model.micro_execute)):
                action = model.micro_execute.pop(0)
                micro_list.append(action)
                action_string = ['a','b','c','d','e','f'][action]
                micro_str += action_string
                tmp_state, step_reward, done, _ = env.step(action)
                if step_reward == -1:
                    step_reward = args.negative_reward
                    micro_str = micro_str[:-1]
                    micro_str += action_string.upper()
                if step_reward == 0:
                    step_reward = args.positive_reward
                if step_reward == 100:
                    step_reward = 5
                reward += step_reward
                if step_reward:  # > -1:
                    # only change the state when make a legal move
                    model.state_changed = True
                    state = tmp_state
                    state = np.array(state)
                    state = torch.from_numpy(state).to(device).unsqueeze(1)
            else:
                reward = 0
                done = None

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                # if len(model.leftmost) == 0:
                    # model.rewards[-1] += 50
                    # ep_reward += 50
                break
            elif len(model.leftmost) == 0:   # if no more unexpanded leftmost non-terminals (the parse tree has fully expanded), but not success
                # model.rewards[-1] -= 50
                # ep_reward -= 50
                break
            # elif t == args.max_steps:
            #     ep_reward -= 2
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward


        if ep_reward >= highest_ep_reward:
            highest_ep_reward = ep_reward

        if ep_reward == 0:
            print(micro_list)

        # perform backprop
        R = 0
        policy_losses = []
        entropy_losses = []
        value_losses = []
        returns = []

        # calculate returns at each timestep
        for r in model.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # calculate losses
        for log_prob, R, action_probs_hist, value in zip(model.action_probs, returns, model.action_probs_hist, model.values):
            entropy = torch.distributions.Categorical(probs=action_probs_hist).entropy()
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)
            entropy_losses.append(- entropy_factor * entropy)
            value_losses.append(F.smooth_l1_loss(value.squeeze(), torch.tensor([R]).squeeze().to(device)))

        # sum up all the losses
        episode_policy_loss = torch.stack(policy_losses).sum()
        episode_entropy_loss = torch.stack(entropy_losses).sum()
        episode_value_loss = torch.stack(value_losses).sum()

        episode_loss = episode_policy_loss + episode_entropy_loss + episode_value_loss

        # with torch.autograd.set_detect_anomaly(True):
        # reset gradients
        optimizer.zero_grad()
        # perform backprop+
        episode_loss.backward()
        optimizer.step()

        # write to tensorboard
        writer.add_scalar('4 Episode reward', ep_reward, i_episode)
        writer.add_scalar('6 Steps per episode', t, i_episode)

        writer.add_scalar('1 Actor loss', episode_policy_loss, i_episode)
        writer.add_scalar('2 Entropy loss', episode_entropy_loss, i_episode)
        writer.add_scalar('3 Critic loss', episode_value_loss, i_episode)

        if done:
            writer.add_scalar('5 Episode success', 1, i_episode)
        else:
            writer.add_scalar('5 Episode success', 0, i_episode)

        writer.add_scalar('7 Number of trees', model.num_trees, i_episode)
        writer.add_scalar('8 Length of terminal sequence', len(micro_list), i_episode)
        # writer.add_scalar('9 Max node count', max_node_count, i_episode)




        # reset rewards and action buffer
        del model.rewards[:]
        del model.action_probs[:]
        model.leftmost = [[(0,1)]]
        model.micro_execute = []
        model.micro_storage = [[]]
        model.hidden_state = None
        del model.action_probs_hist[:]
        del model.values[:]
        model.num_trees = 0


        # log results
        if i_episode % args.log_interval == 0:
            toc = time.time()
            print(
                'Episode {} \t Last reward: {:.2f} \t Average reward: {:.2f} \t Highest reward: {:.2f} \t Time taken: {:.2f}s'.format(
                    i_episode, ep_reward, running_reward, highest_ep_reward, toc - tic))
            #             print(kl.mean())
            tic = toc
            highest_ep_reward = -1e6

        # # check if we have "solved" the cart pole problem
        # if running_reward >= 90:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


        if done:
            success_count += 1
        else:
            success_count = 0

        if i_episode > 4:
            # agent solved the game
            if t < min_step:
                torch.save(model.state_dict(), '/home/xy2419/leftmost-derivation/saved_models/ac_%s_value_%.3f_entropy_%d_nt_%d_pt_%s_decay_%.3f_%.3f_%.3f_reward_%d_disks_%d_steps.pt' 
                %(args.value_one, args.entropy_factor, args.nt_states, args.pt_states, args.decay, args.rule_reward, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps))
                min_step = t
                print('episode %d, step taken %d, micro length %d, model saved' %(i_episode,t,len(micro_list)))

                if i_episode < 20:
                    with open('/home/xy2419/leftmost-derivation/plot/NN_%.3f_entropy_%d_nt_%d_pt_%s_decay_%.3f_%.3f_%.3f_reward_%d_disks_%d_steps.txt' 
                    %(args.entropy_factor, args.nt_states, args.pt_states, args.decay, args.rule_reward, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps), "w") as text_file:
                        text_file.write('graph')
                        text_file.write('\n')
                        text_file.write('{')
                        text_file.write(model.graph)
                        text_file.write('\n')
                        text_file.write('}')
                        text_file.write('\n')
                        text_file.write(micro_str)
                        text_file.write('\n')
                        text_file.write('episode %d, step taken %d, micro length %d' %(i_episode,t,len(micro_list)))
                        text_file.write('\n\n')
                else:
                    with open('/home/xy2419/leftmost-derivation/plot/NN_%.3f_entropy_%d_nt_%d_pt_%s_decay_%.3f_%.3f_%.3f_reward_%d_disks_%d_steps.txt' 
                    %(args.entropy_factor, args.nt_states, args.pt_states, args.decay, args.rule_reward, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps), "a") as text_file:
                        text_file.write('graph')
                        text_file.write('\n')
                        text_file.write('{')
                        text_file.write(model.graph)
                        text_file.write('\n')
                        text_file.write('}')
                        text_file.write('\n')
                        text_file.write(micro_str)
                        text_file.write('\n')
                        text_file.write('episode %d, step taken %d, micro length %d' %(i_episode,t,len(micro_list)))
                        text_file.write('\n\n')

        model.action_hist = []
        model.graph = ''
        model.graph_count = 1

if __name__ == '__main__':
    main()