import deepchem as dc
import torch
from torch_geometric.data import Data, DataLoader, NeighborSampler
import torch.optim as optim
import json

# graph visualize
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx, k_hop_subgraph, degree
import matplotlib.pyplot as plt

# model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


# utils 
import copy
import random
from typing import List
from typing import Dict
from tqdm.auto import tqdm
import time
import argparse 
import os 

#########################
#########################
####### ARGUMENTS #######
#########################
#########################


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=['mwt', 'logp', 'reactivity'])
parser.add_argument("--finetuning",type=str,required=True,choices=['True', 'False'])
parser.add_argument("--batch_size", type=int,required=True)
parser.add_argument("--epochs", type=int,required=True)
parser.add_argument("--data_saving",type=str,required=True,help='')
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--save_dir",type=str ,default='/home/ubuntu/dhkdgmlghks/MLBIO',required=False)
parser.add_argument("--checkpoint_dir",type=str , required=False)
parser.add_argument("--exp_name", type=str, required=True)
args = parser.parse_args()

#######################
#######################
####### MODULES #######
#######################
#######################

## ================== Graph Visualizer ================== ##
def graph_visualizer(data):
  G=to_networkx(data, to_undirected=True)
  nx.draw_networkx(G, with_labels=True)


## ================== Graph Augmentation ================== ## 
class MultiView_generator(object):
  def __init__(self,k_hop):
    self.k_hop = k_hop # k_hop == n_views
    

  def _RandNode_sampler(self) -> int:
    num = random.randrange(self.n_nodes)
    sample_node_idx = num
    return sample_node_idx
  

  def _node_degree(self, data) -> List:
    """sampling node_idx higher than node degree 2 or 3"""
    node_degree = degree(data.edge_index[0], data.num_nodes)
    important_node_idx = list(torch.where(node_degree > 2))[0].tolist() 
    return important_node_idx


  def _ImportantNode_sampler(self,data) -> int:
    important_node_idx = self._node_degree(data)
    num = random.randrange(len(important_node_idx))
    sample_node_idx = important_node_idx[num]
    return sample_node_idx

  
  def _make_subgraph(self, node_idx, k_hop,data):
    n_id, edge_index, _, _, = k_hop_subgraph(node_idx, k_hop, data.edge_index)
    
    node_features = torch.zeros(data.x.size())
    for id in n_id:
      index = id.item()
      node_features[index,:] = data.x[index, :]

    subgraph = Data(x=node_features, edge_index = edge_index, y=data.y )
    return subgraph


  def __call__(self, data) -> Dict:
    subgraphs = {}
    subgraphs['original'] = data
    subgraphs['subgraph'] = []

    #sample_node_idx = self._RandNode_sampler()
    sample_node_idx = self._ImportantNode_sampler(data)
    for i in range(1, self.k_hop + 1):
      subgraph = self._make_subgraph(sample_node_idx, i, data)
      subgraphs['subgraph'] .append(subgraph)
    return subgraphs


## ================== Partitioning datasets ================== ##
class making_datasets(object):
  """
  def _AdjList_to_EdgeIndex(self, adj_list):
    edge_index1 = []
    edge_index2 = []
    for idx, node in enumerate(adj_list):
      edge_index1 = edge_index1 + ([idx] * len(node))
      edge_index2 = edge_index2 + node

    edge_index1 = torch.tensor(edge_index1).unsqueeze(0)
    edge_index2 = torch.tensor(edge_index2).unsqueeze(0)
    edge_index = torch.cat((edge_index1, edge_index2))
    return edge_index
  """
  def _AdjList_to_EdgeIndex(self, adj_list):
    # transforming adjacency list to adjacency matrix 
    num_node = len(adj_list)
    adj_matrix = torch.zeros((num_node,num_node))
    for i,_ in enumerate(adj_list):
      for j in adj_list[i]:
        adj_matrix[i,j] = 1 
    edge_index = adj_matrix.nonzero().t().contiguous()
    return edge_index

  def _making_dataset(self, dataset):
    data = []
    for k in tqdm(range(len(dataset))):
      node_features = torch.tensor(dataset.X[k].get_atom_features())
      edge_index = torch.tensor(self._AdjList_to_EdgeIndex(dataset.X[k].get_adjacency_list()))
      label = torch.tensor(dataset.y[k]).long()
      original_graph = Data(x=node_features, edge_index = edge_index, y =label)

      data.append(original_graph)

    return data

  def __call__(self, dataset):
    dataset = self._making_dataset(dataset)
    return dataset

## ================== Partitioning datasets ================== ##
def loading_labels(dataset, args):
  for data in tqdm(dataset):
    if args.task == 'mwt':
      y_true = copy.copy(data.y[0].item())
      data.y = torch.tensor([y_true])
    elif args.task == 'logp':
      y_true = copy.copy(data.y[1].item())

      data.y = torch.tensor([y_true])
    elif args.task == 'reactivity':
      y_true = copy.copy(data.y[2].item())

      data.y = torch.tensor([y_true])

  return dataset
    


## ================== Model ================== ##
## ====== Backbone ====== ##
# Reference
# https://jaeyong-song.github.io/posts/GNN_Pytorch_Basic/
class GNNStack_backbone(nn.Module):
  def __init__(self, num_layers, input_dim, hidden_dim, drop_out=0.25,task='node'):
    super(GNNStack_backbone, self).__init__()
    self.task = task 
    self.num_layers = num_layers
    self.drop_out = drop_out
    self.hidden_dim = hidden_dim
    # 모듈리스트를 이용해서 레이어 추가
    self.convs = nn.ModuleList()
    self.convs.append(self.build_conv_model(input_dim, hidden_dim))
    # 정규화 레이어 추가
    self.lns = nn.ModuleList()
    self.lns.append(nn.LayerNorm(hidden_dim))
    # 위에서 레이어 1개를 추가했으므로, num_layers-1 만큼 추가
    for l in range(num_layers-1):
      self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
      self.lns.append(nn.LayerNorm(hidden_dim))
    """
    # post-message-passing
    self.post_mp = nn.Seqeuntial(
        nn.Linear(hidden_dim, hidden_dim),
        nn.Dropout(self.drop_out),
        nn.Linear(hidden_dim, output_dim)
    )
    """

  def build_conv_model(self, input_dim, output_dim):
    # refer to pytorch geometric nn module for different implementation of GNNs.
    if self.task == 'node':
      # 원래는 GCNConv를 이용하지만, 아래에서 구현해본 결과를 살펴보기 위해서 해볼 것!
      return pyg_nn.GCNConv(input_dim, output_dim)
    else:
      return pyg_nn.GINConv(nn.Sequential(
          nn.Linear(input_dim, output_dim),
          nn.ReLU(),
          nn.Linear(output_dim, output_dim)
      ))


  def forward(self, data):
    # x has shape [N, in_channels]. 
    # edge_index has shape [2, E]
        
    # Be cautious that Pytorch Geometric concatenate all mini-batch graphs to make one large graph 
    # refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html

    x, edge_index, batch = data.x.float(), data.edge_index, data.batch 
    if data.num_node_features == 0:
      x = torch.ones(data.num_nodes, 1)

    # 각 레이어 사이에는 ReLU와 오버피팅을 막기 위한 드롭아웃 추가
    # 그러나 마지막 레이어에는 추가하지 말아야함... 성능상 이유
    for i in range(self.num_layers):
      x = self.convs[i](x, edge_index)
      emb = x
      x = F.relu(x)
      x = F.dropout(x, p=self.drop_out, training=self.training)
      if not i == self.num_layers -1:
        x = self.lns[i](x)


    # 그래프 분류의 경우에는 mean pool 진행해야함
    if self.task == 'graph':
      x = pyg_nn.global_mean_pool(x, batch)
    
    return emb, x 

    """    
    # 분류를 위한 fc 통과
    x = self.post_mp(x)

    return emb, F.log_softmax(x, dim=1)
    """


## ====== simCLR ====== ##
class simCLR_GCN(nn.Module):
  def __init__(self, backbone, output_dim):
    super(simCLR_GCN,self).__init__()
    self.input_dim = backbone.hidden_dim
    self.backbone = backbone

    # post_message-passing
    self.projection_head = self._projection_head()
    self.fclayer = nn.Linear(in_features=self.input_dim, out_features=output_dim) 

  #def contrastive_loss(self,)
  def _projection_head(self):
    """
    In simCLR V1, there are 2 layers in projection head, and only (frozen) encoder is used to fine-tuning 
    In simCLR V2, there are 3 layers in projection head, and (frozen) encoder and the first (frozen) MLP layer is used to fine-tuning
    """
    head1 = nn.Sequential(
                          nn.Linear(in_features=self.input_dim, out_features=self.input_dim),
                          nn.BatchNorm1d(self.input_dim),
                          nn.ReLU()
                          )
    return head1

  def forward(self, x):
    emb, x = self.backbone(x)
    x = self.projection_head(x)
    x = self.fclayer(x)

    return emb, F.log_softmax(x,1)


## ====== finetuning ====== ##
class finetuning_GCN(nn.Module):
  def __init__(self, backbone, output_dim):
    super(finetuning_GCN, self).__init__()
    self.input_dim = backbone.hidden_dim
    self.backbone = backbone

    # post_message-passing
    self.post_mp = nn.Sequential(
        nn.Linear(self.input_dim, self.input_dim),
        nn.Linear(self.input_dim, output_dim)
    )


  def forward(self, x):
    emb, x = self.backbone(x)
    x = self.post_mp(x)

    return emb, F.log_softmax(x,1)



## ================== Experiments ================== ##
## ====== finetuning ====== ##
def finetuning_train(net, train_loader, optimizer, metric = 'ACC'):
  net.train()

  loss_function = nn.L1Loss()


  
  losses = 0 
  for batch in train_loader:
    optimizer.zero_grad()
    y_true = batch.y
    y_true = y_true.unsqueeze(1)
    y_true = y_true.to(f'cuda:{args.gpu}')
    embedding, pred = net(batch.to(f'cuda:{args.gpu}'))
    loss = loss_function(pred, y_true)
    losses += loss
    loss.backward()
    optimizer.step()

  losses = losses / len(train_loader)
  return net, losses


def finetuning_val(net, val_loader, optimizer, metric='ACC'):
  net.eval()


  loss_function = nn.L1Loss()

    
  losses = 0 
  with torch.no_grad():
    for batch in val_loader:
      y_true = batch.y
      y_true = y_true.unsqueeze(1)
      y_true = y_true.to(f'cuda:{args.gpu}')
      
      embedding, pred = net(batch.to(f'cuda:{args.gpu}'))
      loss = loss_function(pred, y_true)
      losses += loss

  losses = losses / len(val_loader)
  return net, losses


def finetuning_test(net, test_loader, optimizer, metric='ACC'):
  net.eval()

  loss_function = nn.L1Loss()

    
  losses = 0 
  with torch.no_grad():
    for batch in test_loader:
      y_true = batch.y
      y_true = y_true.unsqueeze(1)
      y_true = y_true.to(f'cuda:{args.gpu}')
      embedding, pred = net(batch.to(f'cuda:{args.gpu}'))
      loss = loss_function(pred, y_true)
      losses += loss

  losses = losses / len(test_loader)
  return net, losses


## ================== Utils ================== ##
def metric_calculation(pred, y_true, loss, types):
  if types == 'ACC':
    _, predicted = torch.max(pred.data,1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    ACC = 100 * correct / total
    R2 = None

  else:
    y_var = torch.var(y_true)
    r_square = 1 - (loss / y_var)
    R2 = r_square.item()
    ACC = None

  return ACC, R2


## ================== Optimizer ================== ##
class LAMB(Optimizer):
    """Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(LAMB, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]
                # getting device 
                device = grad.get_device()
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg, exp_avg_sq = exp_avg.to(f'cuda:{device}'), exp_avg_sq.to(f'cuda:{device}')
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                #print(exp_avg.get_device())
                #print(grad.get_device())

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step.to(f'cuda:{p.data.get_device()}'), alpha=-step_size * trust_ratio)

        return loss

###########################
###########################
####### Experiments #######
###########################
###########################

# loading and partitioning dataset
if args.data_saving == 'True':
    tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer='GraphConv')
    train_dataset, val_dataset, test_dataset = datasets

    Dataset = making_datasets()
    train_dataset = Dataset(train_dataset)
    val_dataset = Dataset(val_dataset)
    test_dataset = Dataset(test_dataset)

    torch.save(train_dataset, os.path.join(args.save_dir, 'datasets/train_dataset.pt'))
    torch.save(val_dataset, os.path.join(args.save_dir, 'datasets/val_dataset.pt'))
    torch.save(test_dataset, os.path.join(args.save_dir, 'datasets/test_dataset.pt'))

else:
    print('================== START LOADING DATA ==================')
    train_dataset = torch.load(os.path.join(args.save_dir, 'datasets/train_dataset.pt'))
    val_dataset = torch.load(os.path.join(args.save_dir, 'datasets/val_dataset.pt'))
    test_dataset = torch.load(os.path.join(args.save_dir, 'datasets/test_dataset.pt'))
    
    train_dataset = loading_labels(train_dataset, args)
    val_dataset = loading_labels(val_dataset, args)
    test_dataset = loading_labels(test_dataset, args)
    
    print('================== DONE LOADING DATA ==================')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

backbone = GNNStack_backbone(4, 75, 75, task='graph')
net = finetuning_GCN(backbone, 1)

if args.finetuning == 'True':
  checkpoint = torch.load(args.checkpoint_dir, map_location = 'cpu')
  net.backbone.load_state_dict(checkpoint['backbone'])

  # freezing layers 
  for params in net.backbone.parameters():
    setattr(params, 'requires_grad', False)
    


#net = nn.DataParallel(net)
net.to(f'cuda:{args.gpu}')
#net = simCLR_GCN(backbone, 10)
optimizer = optim.Adam(net.parameters(), lr = 0.01, weight_decay=1e-04)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=10)
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)


result = {}
result['train_loss'] = []
result['val_loss'] = []
result['test_loss'] = []


for epoch in range(args.epochs):
  ts = time.time()
  net, train_loss = finetuning_train(net, train_loader, optimizer, metric = 'R2')
  net, val_loss = finetuning_val(net, val_loader, optimizer, metric = 'R2')
  scheduler.step(val_loss.item())
  te = time.time()
  print('Epoch {}. Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, val_loss.item(),  optimizer.param_groups[0]['lr'],te-ts))
  
  result['train_loss'].append(train_loss.item())
  result['val_loss'].append(val_loss.item())




print('=============== STARTING INFERENCE ===============')
net, test_loss = finetuning_test(net, test_loader, optimizer, metric = 'R2')
print('Epoch {}. Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, test_loss.item(), optimizer.param_groups[0]['lr'],te-ts))

result['test_loss'].append(test_loss.item())


## save results 
result_dir = os.path.join(args.save_dir, 'result')
setting = vars(args)
if os.path.isdir(result_dir) == False:
  os.makedirs(result_dir)

if args.finetuning == 'True':
  checkpoint_file = os.path.split(args.checkpoint_dir)[-1]
  filename = result_dir + '/{}_{}_{}.json'.format(setting['exp_name'], setting['task'], checkpoint_file)
else:
  filename = result_dir + '/{}_{}.json'.format(setting['exp_name'], setting['task'])
result.update(setting)

with open(filename, 'w') as f:
  json.dump(result, f)