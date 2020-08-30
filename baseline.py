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
lr = 1e-3
gamma = 0.99   # discount factor
log_interval = 10   # episode interval between training logs

decay_in = 1e4
seed = 2

# gym-hanoi env settings
num_disks = 3
env_noise = 0.   # transition/action failure probability
state_space_dim = num_disks
action_space_dim = 6   # always 6 actions for 3 poles
env = gym.make("Hanoi-v0")
env.set_env_parameters(num_disks, env_noise, verbose=True)
env.seed(seed)
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

entropy_factor = 0.1  # decay to 0.01
decay = True

positive_reward = 0.01
negative_reward = -0.01

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')

parser.add_argument('--decay', type=bool, default=decay)
parser.add_argument('--decay_in', type=int, default=decay_in)
parser.add_argument('--positive-reward', type=float, default=positive_reward)
parser.add_argument('--negative-reward', type=float, default=negative_reward)

parser.add_argument('--gamma', type=float, default=gamma, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=seed, metavar='N',
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
args = parser.parse_args()



class Policy(nn.Module):
    def __init__(self, action_space=args.action_space_dim,
                z_dim=16,
                state_space=args.state_space_dim*3):
        super(Policy, self).__init__()

        # dimensions
        self.action_space = action_space
        self.state_space = state_space

        # action probs, micro/macro action, reward, hidden state buffer
        self.action_probs = []
        self.action_probs_hist = []
        self.values = []
        self.micro_execute = []  # micro action(s) to execute at this time step
        self.rewards = []

        self.z_dim = z_dim
        self.state_emb_z = nn.Linear(self.state_space, self.z_dim)

        self.rule_mlp = nn.Sequential(nn.Linear(self.z_dim, self.z_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.z_dim, action_space))

        self.values_all = nn.Sequential(nn.Linear(self.z_dim, self.z_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.z_dim, 1))

    def forward(self, state, use_mean=True):
        assert len(self.micro_execute) == 0   # no unexecuted micro-actions

        # change state to one hot vector
        state_one_hot = torch.FloatTensor(int(self.state_space/3), 3).zero_().to(device)
        state_emb = state_one_hot.scatter(1, state, 1).view(1, -1)    # 1 x (3 x state_space)

        self.z = F.relu(self.state_emb_z(state_emb))
        z = self.z
       
        action_score = self.rule_mlp(z).squeeze()
        action_prob = F.softmax(action_score)

        m = Categorical(action_prob)
        action = m.sample()
        log_prob = m.log_prob(action)

        values = self.values_all(z)
        value = values[:]

        a = int(action)
        self.micro_execute.append(a)

        # return log_prob for the chosen rule (for policy loss calculation), and return rule prob for all rules that can be chosen at this step (for entropy calculation)
        return log_prob, action_prob, value


def linear_anneal(counter, start, final, in_steps):
    out = max(final, start + counter * (final-start) / in_steps)
    return out


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = Policy().to(device)
    # if args.load:
    #     model.load_state_dict(torch.load(args.load_path))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    model_name = 'baseline_%.3f_entropy_%s_decay_%.3f_%.3f_reward_%d_disks_%d_steps_%d_seed'\
        %(args.entropy_factor, args.decay, args.positive_reward, args.negative_reward, args.state_space_dim, args.max_steps, args.seed)
    
    writer = SummaryWriter(log_dir='/home/xy2419/leftmost-derivation/runs/'+model_name+'/')

    tic = time.time()
    running_reward = 0
    highest_ep_reward = -1e6   # record the highest episode reward in each log-interval

    success_count = 0
    entropy_factor = args.entropy_factor
    min_step = 1e5
    for i_episode in range(1, int(args.max_episodes) + 1):  # count(1):
        if args.decay:
            entropy_factor = linear_anneal(counter=i_episode, start=args.entropy_factor, final=0.01, in_steps=args.decay_in)

        micro_list = []   # store the list of action sequence for each episode, in case we need to check
        micro_str = ''

        # reset environment and episode reward
        state = env.reset()
        state = np.array(state)
        state = torch.from_numpy(state).to(device).unsqueeze(1)
        ep_reward = 0

        for t in range(1, int(args.max_steps) + 1):    # tqdm(range(1, max_steps)):
            # select action from policy
            log_prob, action_prob, value = model(state)
            model.action_probs.append(log_prob)
            model.action_probs_hist.append(action_prob)
            model.values.append(value)

            # reward = args.rule_reward
            assert len(model.micro_execute) > 0
               
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
            reward = step_reward

            # only change the state when make a legal move
            model.state_changed = True
            state = tmp_state
            state = np.array(state)
            state = torch.from_numpy(state).to(device).unsqueeze(1)
           
            model.rewards.append(reward)
            ep_reward += reward

            if done:
                # if len(model.leftmost) == 0:
                    # model.rewards[-1] += 50
                    # ep_reward += 50
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if ep_reward >= highest_ep_reward:
            highest_ep_reward = ep_reward


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

        # writer.add_scalar('7 Number of trees', model.num_trees, i_episode)
        writer.add_scalar('8 Length of terminal sequence', len(micro_list), i_episode)
        # writer.add_scalar('9 Max node count', max_node_count, i_episode)


        # reset rewards and action buffer
        del model.rewards[:]
        model.micro_execute = []
        del model.action_probs_hist[:]
        del model.action_probs[:]
        del model.values[:]


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