<div align=center><img src="./doc/figure2.jpg"></div>

# DYGL-lib

代码复现困难和实验方法不一致阻碍了动态网络领域的发展。 我们提出了 DYGL 库，这是一个用于动态图形表示学习的统一、全面且可扩展的库。 该库的主要目标是使研究人员可以在统一且易于使用的框架中进行动态图形表示学习。 为了加速新模型的开发，我们基于统一的数据格式设计了统一的模型接口，有效封装了实现的细节。 实验证明了库中实现的模型在节点分类和链路预测方面的预测性能。 我们的库将有助于动态图领域的标准化和可再现性。
DYGL 目前支持**动态图表示提取、链接预测**和**节点分类**任务。。

## Overall Framework

***Configuration Module**:负责管理框架中的涉及的所有参数。

***Data Module**:负责下载数据集、对数据集进行预处理、保存和加载数据集。

***Model Module**:负责初始化基线模型或自定义模型。

***Evalution Module**:提供统一的下游任务评测模型，并通过多个指标评估算法性能。

## GET STARTED

### Install and Setup

lib只能从源代码安装

```shell

git clonehttps://github.com/half-salve/DYGL-lib

cd  DYGL-lib

```

### Configure Dependencies

After obtaining the source code, you can configure the environment.

Our code is based on Python version 3.7 and PyTorch version 1.12.0+. For example, if your CUDA version is 11.6, you can install PyTorch with the following command.

在获得源代码后，你可以开始构建管理你的代码环境。

我们的代码基于python3.7和pytorch1.13.0 构建，我们在pytorch1.7的环境下依然可以运行，如果你的cuda版本支持11.6，你可以通过以下的命令创建环境

conda:

```sh

conda installpytorchtorchvisiontorchaudiopytorch-cuda=11.6-cpytorch-cnvidia

conda install-cdglteam/label/cu116dgl

```

### Quick start

在运行模型之前，确保相关的代码和lib在同级目录下，lib的数据集会被处理成dgl.graph的数据格式储存在DYGL_dataset目录下。

脚本'run_model.py'用于在lib中训练和评估单个模型。运行'run_model.py'时，必须指定以下三个参数，即

**task、dataset和model**。在参数中可以选择**gpu**的编号，如果不指定的话，会检测是否有可以使用的显卡，如果有可以使用的显卡，会使用默认使用**编号0**的显卡，如果没有可使用的显卡，会使用**cpu**来运行模型。例如：

```sh

python run_model.py--tasklink_prediction--modelTGN--datasetwikipedia--gpu0

```

该脚本在默认配置下，在wikipeida数据集上通过gpu 0运行TGN模型，以进行链接预测任务。

### Visualization

在模型的训练过程中，我们会记录每个epoch训练过程中loss的值和训练过程中和训练后的评估结果，他们通过.npy格式储存在 `./metric`文件夹中。

模型运行一次后，可以通过使用以下命令进行可视化：

```sh


```

### Reproduced Model List

lib中所复现的全部模型列表，他们的简称和相关论文如下

| source    | model | title                                                                             |

|-----------|-------|-----------------------------------------------------------------------------------|

| ICLR 2019 | dyrep | Learning Representation over Dynamic Graph                                        |

| KDD 2019  | Jodie | Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks          |

| ICLR 2020 | TGAT  | Inductive Representation Learning on Temporal Graphs                              |

| ICLR 2020 | TGN   | Temporal Graph Networks for Deep Learning on Dynamic Graphs                       |

| ICLR 2021 | CAW   | INDUCTIVE REPRESENTATION LEARNING IN TEMPORAL NETWORKS VIA CAUSAL ANONYMOUS WALKS |

## API REFERENCE

### lib.config

[DGCONFIG(Base Class)](./doc/config/ConfigParser.md) | [ConfigParser](./doc/config/ConfigParser.md)

### lib.data

#### [DYGLDataset(Base Class)](./doc/data/Base_class.md)

#### Link Prediction Datasets

[RedditData](./doc/data/Reddit.md) | [WikipediaDataset](./doc/data/wikipedia.md)

#### Link state change Prediction Datasets

[RedditDataset](./doc/data/Reddit.md) | [WikipediaDataset](./doc/data/wikipedia.md)

#### [lib.data.Utilities](./doc/data/data.utils.md)

### lib.dataloading

#### [RandEdgeSampler](./doc/dataloading/RandEdgeSampler.md) | [NeighborFinder](./doc/dataloading/NeighborFinder.md)

### lib.nn

[CAWconv](./doc/nn/CAWConv.md) | [TGNconv](./doc/nn/TGNConv.md) | [TGAT](./doc/nn/TGATConv.md) | [Jodie](./doc/nn/JodieConv.md) | [DyRep](./doc/nn/derep.md)

### lib.evaluator

#### task for link prediction

[MergeLayer](./doc/evalution/edge_aggregation.md) | [link_prediction_metric](./doc/evalution/edge_aggregation.md)

#### task for node classification

[MLP](./doc/evalution/node_class.md)
