# lib

lib 目前支持**动态图表征提取、链接预测**和**节点分类**任务。

## Overall Framework

* **Configuration Module**:负责管理框架中的涉及的所有参数。
* **Data Module**:负责下载数据集、对数据集进行预处理、保存和加载数据集。
* **Model Module**:负责初始化基线模型或自定义模型。
* **Evalution Module**:提供统一的下游任务评测模型，并通过多个指标评估算法性能。

## GET STARTED

### Install and Setup

lib只能从源代码安装

```shell
git clone https://github.com/half-salve/DYGL-lib
cd  DYGL-lib
```

### Quick start

在运行模型之前，确保相关的代码和lib在同级目录下，lib的数据集会被处理成dgl.graph的数据格式储存在DYGL_dataset目录下。

脚本'run_model.py'用于在lib中训练和评估单个模型。运行'run_model.py'时，必须指定以下三个参数，即
**task、dataset和model**。在参数中可以选择**gpu**的编号，如果不指定的话，会检测是否有可以使用的显卡，如果有可以使用的显卡，会使用默认使用**编号0**的显卡，如果没有可使用的显卡，会使用**cpu**来运行模型。例如：

```sh
python run_model.py --task link_prediction --model TGN --dataset wikipedia --gpu 0
```

该脚本在默认配置下，在wikipeida数据集上通过0号gpu运行TGN模型，以进行链接预测任务。**lib中数据集、模型和任务之间的对应关系表格如下**

### Visualization

在模型的训练过程中，我们会记录每个epoch训练过程中loss的值和训练过程中和训练后的评估结果，他们通过.npy格式储存在 `./metric`文件夹中。
模型运行一次后，可以通过使用以下命令进行可视化：

```sh

```

### Reproduced Model List

lib中所复现的全部模型列表，他们的简称和相关论文如下，

## API REFERENCE

### lib.config
[DGCONFIG](./doc/config/ConfigParser.md) | [ConfigParser](./doc/config/ConfigParser.md)

### lib.data

#### [DYGLDataset(Base Class)](./doc/data/Base_class.md)

#### Link Prediction Datasets

[RedditData](./doc/data/Reddit.md) | [WikipediaDataset](./doc/data/wikipedia.md)

#### Link state change Prediction Datasets

[RedditDataset](./doc/data/Reddit.md) | [WikipediaDataset](./doc/data/wikipedia.md)

#### [lib.data.Utilities]()

### lib.dataloading
#### [RandEdgeSampler](./doc/dataloading/RandEdgeSampler.md) | [NeighborFinder](./doc/dataloading/NeighborFinder.md)

### lib.evaluator

### lib.nn

### lib.utils
