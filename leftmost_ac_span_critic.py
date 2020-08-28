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

max_steps = 500    # max steps the agent can execute per episode
max_episodes = 15000  # max number of episodes
entropy_factor = 0.1  # coefficient multiplying the entropy loss
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
        self.value_network = nn.Sequential(nn.Linear(state_space, 32),
                                            ResidualLayer(32, 32),
                                            ResidualLayer(32, 32),
                                            nn.Linear(32, 1))
        self.state_space = state_space
    
    def forward(self, state):
        # change state to one hot vector
        state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
        state_emb = state_one_hot.scatter(1, state, 1).view(1, -1)    # 1 x (3 x state_space)

        value = self.value_network(state_emb)
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
        self.action_probs = []
        self.action_probs_hist = []
        # self.values = []
        self.leftmost = [[0]]   # leftmost unexpanded non-terminal, append a list of non-terminal(s) for each expansion
        self.micro_execute = []  # micro action(s) to execute at this time step
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
        self.nt_rule_mlp = nn.Linear(symbol_dim + self.z_dim, self.nt_nt_rule_num)
        self.pt_rule_mlp = nn.Sequential(nn.Linear(symbol_dim + self.z_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      ResidualLayer(symbol_dim, symbol_dim),
                                      nn.Linear(symbol_dim, action_space))

    def forward(self, nt, state, use_mean=True):
        assert len(self.micro_execute) == 0   # no unexecuted micro-actions

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


        # pop the leftmost non-terminal

        # nt = self.leftmost[-1].pop(0)
        # if len(self.leftmost[-1]) == 0:
        #     del self.leftmost[-1]
        # assert nt == 0 or nt >= self.action_space   # assert nt is either the root or nt/pt, not a terminal
        # assert len(self.leftmost) == len(self.micro_storage)   # append a list of micros/macros at each hierarchy even if there're none, so they should be of same length

        # calculate rule probs
        if nt == 0:

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

            self.leftmost.append([nt1,nt2])

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
                nt_rule_score = self.nt_rule_mlp(nt_emb).squeeze()
                rule_prob = F.softmax(nt_rule_score)

                # select expansion rule
                m = Categorical(rule_prob)
                rule = m.sample()
                log_prob = m.log_prob(rule)

                nt1 = int(rule//(self.pt_states + self.nt_states) + self.action_space)
                nt2 = int(rule%(self.pt_states + self.nt_states) + self.action_space)

                self.leftmost.append([nt1,nt2])

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

                self.micro_execute.append(a)

        # return log_prob for the chosen rule (for policy loss calculation), and return rule prob for all rules that can be chosen at this step (for entropy calculation)
        return log_prob, rule_prob


def calculate_return(reward_list, gamma):
    reward_list = [item for sublist in reward_list for item in sublist]

    discounted_return = 0
    for R in reversed(reward_list):
        discounted_return = R + gamma*discounted_return
    return discounted_return



def main():
    model = Policy().to(device)
    critic = Critic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    writer = SummaryWriter(log_dir='/home/xy2419/leftmost-derivation/runs/span_%.3f_entropy_%d_nt_%d_pt/' %(args.entropy_factor, args.nt_states, args.pt_states))

    tic = time.time()
    running_reward = 0
    highest_ep_reward = -1e6   # record the highest episode reward in each log-interval

    success_count = 0
    episode_entropy_loss = []
    episode_actor_loss = []
    episode_critic_loss = []
    entropy_factor = args.entropy_factor
    for i_episode in range(1, args.max_episodes + 1):  # count(1):
        # if i_episode > 200 and i_episode < 400:
        #     entropy_factor /= 2
        # elif i_episode > 400 and i_episode < 600:
        #     entropy_factor /= 2
        # elif i_episode > 600 and i_episode < 800:
        #     entropy_factor /= 2
        # elif i_episode > 800 and i_episode < 1000:
        #     entropy_factor /= 2
        # elif i_episode > 1000:
        #     entropy_factor /= 2

        micro_list = []   # store the list of action sequence for each episode, in case we need to check
        value_list = []
        reward_list = []

        # reset environment and episode reward
        state = env.reset()
        state = np.array(state)
        state = torch.from_numpy(state).to(device).unsqueeze(1)

        nt = 0
        model.leftmost.append([])
        ep_reward = 0
        for t in range(1, args.max_steps + 1):    # tqdm(range(1, max_steps)):
            if t > 1:
                # if not at the first timestep, the state may not have changed from the previous step
                model.state_changed = False

            old_value = critic(state)
            value_list.append([old_value])

            # select action from policy
            log_prob, rule_prob = model(nt, state)
            model.action_probs.append(log_prob)
            model.action_probs_hist.append(rule_prob)

            reward = 0
            advantage_critic = torch.tensor([0.]).cuda().reshape(1,1)
            new_value = critic(state)

            if len(model.micro_execute) > 0:
                # for _ in range(len(model.micro_execute)):
                action = model.micro_execute.pop(0)
                micro_list.append(action)
                tmp_state, step_reward, done, _ = env.step(action)
                if step_reward == -1:
                    step_reward = -0.01
                if step_reward == 0:
                    step_reward = 0.01
                if step_reward == 100:
                    step_reward = 5 * 2
                reward += step_reward

                state = tmp_state
                state = np.array(state)
                state = torch.from_numpy(state).to(device).unsqueeze(1)

                # execute a non-terminal, update that step immediately 
                # as the pre-terminal is fully expanded, do not store value/action prob, only store reward
                advantage_critic += reward + args.gamma * new_value - old_value
                # critic_loss = advantage_critic ** 2
                # optimizer_critic.zero_grad()
                # critic_loss.backward()
                # optimizer_critic.step()

            else:
                reward = 0
                done = None

            ep_reward += reward
            if len(reward_list) < len(model.leftmost):
                reward_list.append([reward])
            elif len(reward_list) == len(model.leftmost):
                reward_list[-1].append(reward)
            else:
                print('''something's wrong, length of reward_list > model.leftmost''')
            
            for i,(nt_list,value) in enumerate(zip(reversed(model.leftmost),reversed(value_list))):
                if nt_list == [] and value != []:
                    assert len(value) == 1
                    advantage_critic += calculate_return(reward_list[-(i+1):],args.gamma) + args.gamma**(i+1) * new_value - value[0]
                    value_list[-(i+1)] = []
                else:
                    break

            
            print(t)
            
            if advantage_critic != 0:
                critic_loss = advantage_critic**2
                episode_critic_loss.append(critic_loss)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()
            else:
                episode_critic_loss.append(advantage_critic)


            entropy = torch.distributions.Categorical(probs=rule_prob).entropy()
            entropy_loss = - entropy_factor * entropy
            episode_entropy_loss.append(entropy_loss.item())

            advantage_actor = reward + args.gamma * new_value - old_value
            actor_loss = - log_prob * advantage_actor.detach()
            episode_actor_loss.append(actor_loss.item())
            optimizer.zero_grad()
            (actor_loss+entropy_loss).backward()
            optimizer.step()


            if done:
                break
            else:
                nt = None
                for i,nt_list in enumerate(reversed(model.leftmost)):
                    if nt_list != []:
                        nt = model.leftmost[-(i+1)].pop(0)
                if nt == None:
                    nt = 0
                    model.leftmost = [[]]
                    value_list = []
                    reward_list = []



        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if running_reward > 10:
            success_count += 1
        else:
            success_count = 0

        if success_count > 200:
            # agent solved the game
            torch.save(model.state_dict(), '/home/xy2419/leftmost-derivation/saved_models/span_%.3f_entropy_%d_nt_%d_pt_%s_lstm.pt' %(args.entropy_factor, args.nt_states, args.pt_states, args.use_lstm))
            print('Agent solved the game!')
            break

        if ep_reward >= highest_ep_reward:
            highest_ep_reward = ep_reward

        if ep_reward == 0:
            print(micro_list)

      
        # write to tensorboard
        writer.add_scalar('Episode reward', ep_reward, i_episode)
        writer.add_scalar('Steps per episode', t, i_episode)

        writer.add_scalar('Policy loss', np.mean(episode_actor_loss), i_episode)
        writer.add_scalar('Entropy loss', np.mean(episode_entropy_loss), i_episode)
        # writer.add_scalar('Critic loss', torch.mean(episode_critic_loss), i_episode)

        if done:
            writer.add_scalar('Episode success', 1, i_episode)
        else:
            writer.add_scalar('Episode success', 0, i_episode)

        if len(model.leftmost) == 0:
            # parse tree fully expanded
            writer.add_scalar('Parse tree fully expanded', 1, i_episode)
        else:
            writer.add_scalar('Parse tree fully expanded', 0, i_episode)

        # reset rewards and action buffer
        # del model.rewards[:]
        del model.action_probs[:]
        model.leftmost = [[0]]
        model.micro_execute = []
        model.micro_storage = [[]]
        model.hidden_state = None
        del model.action_probs_hist[:]
        # del model.values[:]

        # log results
        if i_episode % args.log_interval == 0:
            toc = time.time()
            print(
                'Episode {} \t Last reward: {:.2f} \t Average reward: {:.2f} \t Highest reward: {:.2f} \t Time taken: {:.2f}s'.format(
                    i_episode, ep_reward, running_reward, highest_ep_reward, toc - tic))
            #             print(kl.mean())
            tic = toc
            highest_ep_reward = -1e6


if __name__ == '__main__':
    main()