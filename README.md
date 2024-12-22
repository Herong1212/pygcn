Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](figure.png)

Note: There are subtle differences between the TensorFlow implementation in https://github.com/tkipf/gcn and this PyTorch re-implementation. This re-implementation serves as a proof of concept and is not intended for reproduction of the results reported in [1].

This implementation makes use of the Cora dataset from [2].

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```

## self-add
文件结构分析
1. pygcn
1.1 __init__.py
功能:
    将 pygcn 定义为一个 Python 包，便于导入使用

1.2 layers.py
功能：
    该文件定义了 GCN 的核心组件 GCNLayer（图卷积层）。
    实现了图卷积的核心计算，包括通过邻接矩阵进行邻居节点的特征聚合。
    
1.3 models.py
功能：
    定义了整个 GCN 模型的结构（如前向传播、卷积层实现），包含多层图卷积。

1.4 train.py
功能：
    实现模型的训练和测试逻辑。训练脚本，调用模型、加载数据并进行训练。
    
1.5 utils.py
功能：
    提供了数据处理和预处理的辅助函数，包括加载数据、构建邻接矩阵、数据预处理和特征矩阵。
