# Contrastive-graph-representation-learning-via-graph-subsampling

## Project Task 
_**Hierarchical molecular graph representation learning via contrastive loss-based self-supervised learning.**_

## Objective and Motivation 
이 연구에서는 라벨없이 데이터만으로 학습하는 모델인 self supervised learning을 활용하여 molecular graph의 heirarchical한 특징을 잘 학습할 수 있는지를 확인해보고자 한다.  
이는 라벨과 연관이 있는 molecular graph의 heirarchical한 특징을 학습하기보다는 molecular graph가 가지고 있는 그래프 구조 상의 heirarchical한 특징을 모델이 학습할 수 있게 해줄 것이며, 이렇게 pre-training 된 모델은 이후의 prediction task의 성능을 높여주는 데에 도움을 줄 것이라는 가설을 세우고자 한다.  

## Background 
_**Subgraph Sampling for Graph Neural Networks**_: Subgraph sampling은 다양한 이유로 graph neural network에 쓰인다.  
크게는 그래프 데이터를 처리하는 데에 드는 memory와 같은 computational cost를 줄이면서도 그래프 데이터를 잘 학습시키고자 하는 접근인 SAGE류의 방법들이 있고[[1]], 또 한편으로는 그래프 데이터가 가지고 있는 hierarchical한 특성을 잘 학습하는 모델을 만들고자 하는 방법으로 DiffPool이 있다.[[2]]  
우리의 접근은 DiffPool의 접근과 유사하게 graph의 hierarchical한 특징을 잘 학습하는 모델을 만들고자 한다.  
Molecular는 각각의 원소들이 연결되면서 원래 원소들이 갖고 있던 특성과는 전혀 다른 특성을 가지게 된다.  
이렇게 molecular는 hierarchical한 특징을 갖고 있다는 점에서, 이를 잘 학습하는 모델을 만드는 것이 중요하다.  
이를 위해서 subgraph sampling은 하나의 주요한 방법으로로 사용될 수 있다.  
  
_**Contrastive Loss**_: 최근 Computer Vision에서 self-supervised learning의 방식으로 활발하게 개발되고 있는 것은 data augmentation 방법을 이용하는 방식이다.  
Data augmentation에 근거한 self-supervised 모델들의 핵심은 같은 이미지에 augmentation을 적용하여 변형된 이미지들은 서로 같은 것으로, 서로 다른 이미지에 augmentation을 적용하여 변형된 이미지들은 서로 다른 것으로 학습하는 모델을 만드는 것이다.  
이러한 접근으로 모델을 학습시키는 것에서 가장 중요한 요소는 어떤 augmentation을 어느 정도로 이미지에 적용하여 변형시킴으로써, 같은 이미지에서 변형된 이미지는 같은 것으로, 다른 이미지에서 변형된 이미지는 다른 것으로 학습하는 task의 난이도를 조절하는 것이다.  
simCLR 논문에 따르면 전체 이미지의 일부를 crop하고 rotation과 color jittering을 같이 적용하는 것이 가장 좋은 성능을 보여주는 것으로 보고 되었다.[[3]]   
Graph Augmentation and Contrastive-Based Self-Supervised Learning for Graph: Graph 데이터에서도 다양한 augmentation의 방법이 개발되어 있다. 
Node 혹은 Edge feature를 변형하는 방식의 augmentation[[4], [5]]도 있고, 임의의 edge를 drop하거나[[6]] 임의의 노드로부터 k-hop 떨어진 node들만으로 subgraph를 만들어서 GNN을 학습시키는 방식도 있다. 
Contrastive-Based Self-Supervised Learning에서는 node와 edge augmentation을 기반으로 한 contrastive learning을 사용하는 모델들이 있다.[[7]]  
Sub graph를 샘플링하는 것을 augmentation의 방법으로 활용하여 contrastive learning을 적용하는 연구 또한 존재한다.[[8]]  
우리의 연구가 이 연구와 차별화되는 지점은, 우리 연구는 k-hop sub graph sampling 방식을 사용하였으며, degree of node가 2 혹은 3이상인 노드를 중요 노드로 보고 이 노드를 중심으로 하여 샘플링한 sub graph들 간의 Contrastive loss를 minimize하는 방식으로 해결책을 제시한다는 점이다.  
이는 degree of node가 2 혹은 3이상인 노드는 서로 다른 원소들이 결합하는 지점이자, 개별 원소들의 특징과 전혀 다른 결합물의 특징을 결정 짓는 데에 중요한 역할을 할 것이라고 생각했기 때문이다.  
또한 중요 노드로부터 서로 다른 k의 k-hop sub graph sampling을 진행하여, local graph와 global graph 간의 비교를 통해서 모델이 그래프의 hierarchical한 특징을 포착할 수 있게 하였다는 점에서 차이가 존재한다.  

## Methods 
![alt text](http://url/to/img.png)

_**Loss**_: Contrastive Learning에서 많이 쓰이는 InfoNCELoss를 사용한다.[[3]]
기존의 방법들은 하나의 이미지에서 augmentation을 적용하여 두개의 이미지를 만들고, 같은 이미지들로부터 나온 이미지는 같은 것으로 다른 이미지들로부터 나온 이미지는 다른 것으로 구분할 수 있도록 loss를 구성하였는데, 우리 연구의 목적은 Contrastive learning의 방식을 통해 molecular graph의 hierarchical한 특징을 학습하는 것이 목표이기 때문에, augmentation된 데이터를 비교하는 것이 아니라 original graph와 subgraph를 비교하는 방식으로 loss를 구성하였다.  
또한 molecular graph의 hierarchical한 특징을 학습하게할 수 있도록, original graph로부터 서로 다른 k의 k-hop subgraph를 샘플링 하였다.  
이때 sampling된 subgraph들은 original graph와의 비교만을 하도록 loss를 구성하였다.  
```  
z_i= encoder(subgraph_i),,i ≥ 1 
z_orig= encoder(original graph) 
Loss =1/k *  Σ InfoNCELoss(z_i,z_orig),k ≥ i ≥ 1 
```

_**Graph Subsampling**_: Molecular graph의 경우, 같은 molecule이라면 graph의 structure는 isomorphic하기 때문에, subgraph를 샘플링하는 과정에서는 임의의 initial node를 설정하고, initial node로부터 k-hop만큼 떨어져있는 subgraph를 샘플링하도록 하였다.  
또한, 서로 다른 기본단위 원소들이 결합되는 위치에 해당하는 node, 즉 degree of node가 3이상 되는 node를 initial node로 설정하게 되면, 서로 다른 기본 단위 원소들이 결합됨으로써 형성되는 새로운 특징들을 파악할 수 있다고 생각하였다.  
따라서 degree of node가 3이상 되는 node를 initial node로 설정하였다. 

 
[1]: https://arxiv.org/abs/1810.00826
[2]: https://papers.nips.cc/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html
[3]: https://proceedings.mlr.press/v119/chen20j.html
[4]: https://bhooi.github.io/papers/nodeaug_kdd20.pdf
[5]: https://people.ece.umn.edu/users/ayasin/Publications.html
[6]: https://arxiv.org/abs/1907.10903
[7]: https://arxiv.org/abs/2103.00111
[8]: https://arxiv.org/pdf/2009.10273.pdf
