import numpy as np
import torch
import argparse


# action_space_dim = 6
# nt = 6    
# pt = 6 

# parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
# parser.add_argument('--action-space-dim', type=int, default=action_space_dim, metavar='N',
#                     help='action space dimension (default: 6)')
# parser.add_argument('--nt-states', type=int, default=nt, metavar='N',
#                     help='number of non-terminals (default: 6)')
# parser.add_argument('--pt-states', type=int, default=pt, metavar='N',
#                     help='number of pre-terminals (default: 6)')
# args = parser.parse_args()

# class Tree:
#     def __init__(self, parent_tree, 
#                 left_node=None, 
#                 right_node=None, 
#                 step=None,
#                 action_space=args.action_space_dim,
#                 nt_states=args.nt_states,
#                 pt_states=args.pt_states):
#         self.parent_tree = parent_tree
#         self.left_node = left_node
#         self.right_node = right_node

#         self.action_space = action_space
#         self.nt_states = nt_states
#         self.pt_states = pt_states

#         self.left_node_type = self.node_type(left_node)
#         self.right_node_type = self.node_type(right_node)
#         self.left_tree = None
#         self.right_tree = None

#     def node_type(self,node):
#         if node < self.action_space:
#             return 'T'
#         elif node < self.action_space + self.nt_states:
#             return 'NT'
#         else:
#             return 'PT'

#     def expanded(self):
#         if isinstance(self.left_tree, Tree) and isinstance(self.right_tree, Tree):
#             # parent=NT/S, and both children are Trees
#             if (self.left_tree.expanded() == True) and (self.right_tree.expanded() == True):
#                 return True
#             else:
#                 return False
#         elif self.left_node_type == 'T':
#             # parent=PT, only one children T
#             return True
#         else:
#             return False

# reward_list = []
# value_list = []
# action_logprob_list = []
# action_dist_list = []

# # parent_tree = Tree('S', left_node=None, right_node=None, step=0)

# parent_node = 'S'
# parent_tree = None

# step = 0
# state_old = 0   # state of parent node
# value_old = 0.1   # value of parent node
# value_list.append(oldvalue)
# # run model
# reward = -0.01  # reward of this expansion + any terminal execution
# action_logprob = -5  # log prob of the chosen action
# action_dist = np.array([0,0.1,0.2,0.7])   # distribution of the action space
# children_list = []

# if len(children_list) == 1:
#     # arrive at terminal, execute
#     child = children_list[0]
#     assert child < args.action_space_dim  # assert child is a terminal action
#     reward_terminal = 0.01
#     state = 1
#     done = None
#     rewad += reward_terminal


# if parent_node ==  'S':
#     child_tree = Tree(parent_node, nt1, nt2, step=step)
# else:
#     child_tree = Tree(parent_tree, nt1, nt2, step=step)
#     if parent_tree.left_tree is None:
#         parent_tree.left_tree = child_tree
#     else:
#         parent_tree.right_tree = child_tree

# reward_list.append(reward)
# action_logprob_list.append(action_logprob)
# action_dist_list.append(action_dist)


a = [1,2,3,4,5]
print(a[1,2])
