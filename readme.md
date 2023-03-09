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
git clone 
cd
```

### Quick start

在运行模型之前，确保相关的代码和lib在同级目录下，lib的数据集会被处理成dgl.graph的数据格式储存在DYGL_dataset目录下。

脚本'run_model.py'用于在lib中训练和评估单个模型。运行'run_model.py'时，必须指定以下三个参数，即
**task、dataset和model**。**GPU**如果不指定的话，会检测是否有可以使用的显卡，如果有可以使用的显卡，会使用
编号**0**的显卡，如果没有可使用的显卡，会使用**cpu**来运行模型。

```sh
python run_model.py --task link_prediction --model TGN --dataset wikipedia
```

该脚本在默认配置下，在wikipeida数据集上运行TGN模型，以进行链接预测任务。**lib中数据集、模型和任务之间的对应关系表格如下**

### Visualization

在模型的训练过程中，我们会记录每个epoch训练过程中loss的值和训练过程中和训练后的评估结果，他们通过.npy格式储存在 `./metric`文件夹中。
模型运行一次后，可以通过使用以下命令进行可视化：

```sh

```

### Reproduced Model List

lib中所复现的全部模型列表，他们的简称和相关论文如下，

## API REFERENCE

### lib.config

### lib.data

#### Base Class

##### DYGLDataset

```
class dgl.data.DGLDataset(name, url=None, raw_dir=None, save_dir=None, hash_key=(), force_reload=False, verbose=False, transform=None)
```

Base : `Object`
这是一个基础的数据集类用于创建动态图数据集，它将会通过以下步骤完成初始化,它的定义我们参考了dgl的基类数据集DGLDataset：

1. 检查该数据集是否已经存在（已经被处理好并存放在硬盘中）。如果是转到5。

2. 如果`url`不是None,调用`download()`下载数据。

3. 调用`process()`来处理数据。

4. 调用`save()`将已经处理好的数据保存到硬盘上，然后转到6

5. 用`load()`将已经处理好的数据集从硬盘上加载出来

6. 完成

    用户可以用自己的数据处理逻辑覆盖这些函数。

参数：
* name (str) – 数据集的名称

* url (str) – 下载原始数据集的 URL。 默认值：无

* raw_dir (str) – 指定将存储下载数据的目录或已存储输入数据的目录。 默认值：`os.getwd()/DYGL_dataset/`

* save_dir (str) – 保存已处理数据集的目录。 默认值：与 raw_dir 相同

* hash_key (tuple) – 作为哈希函数输入的值元组。 用户可以通过比较哈希值来区分来自同一数据集类的实例（及其在磁盘上的缓存）。 默认值：()，对应的哈希值为'f9065fa7'。

* force_reload (bool) – 是否重新加载数据集。 默认值：假


#### Link Prediction Datasets

##### RedditDataset
```
class dgl.data.RedditDataset(raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0)
```

Base : `lib.data.JODIEDataset`

Reddit Dataset Statistics

* Nodes: 11,000
* Edges: 672,447
* Node feature size: 172
* Nodes with dynamic labels
* Feature type: LIWC category vector

`__getiem__(idx)`
参数 ：**idx**( *int* ) 
返回 ：这个动态图包括
* `edata['time']`:
* `edata['train_edge_mask']`: positive training edge mask for **Transductive task**
* `edata['val_edge_mask']`: positive validation edge mask for **Transductive task**
* `edata['test_edge_mask']`: positive testing edge mask for **Transductive task**

* `edata['train_edge_observed_mask']`: positive training edge mask for **Inductive task**
* `edata['valid_edge_observed_mask']`: positive validation edge mask for **Inductive task**
* `edata['test_edge_observed_mask']`: positive testing edge mask for **Inductive task**
    
* `features` : Node features
* `edge_feat` : edge features
* `state` :  edge state change labels


##### WikipediaDataset
```
class dgl.data.WikipediaDataset(raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0)
```

Base : `lib.data.JODIEDataset`

Reddit Dataset Statistics

- Nodes: 9,227
- Edges: 157,474
- Node feature size: 172
- Nodes with dynamic labels
- Feature type: LIWC category vector

`__getiem__(idx)`
参数 ：**idx**( *int* ) 
返回 ：这个动态图包括
* `edata['time']`:
* `edata['train_edge_mask']`: positive training edge mask for **Transductive task**
* `edata['val_edge_mask']`: positive validation edge mask for **Transductive task**
* `edata['test_edge_mask']`: positive testing edge mask for **Transductive task**

* `edata['train_edge_observed_mask']`: positive training edge mask for **Inductive task**
* `edata['valid_edge_observed_mask']`: positive validation edge mask for **Inductive task**
* `edata['test_edge_observed_mask']`: positive testing edge mask for **Inductive task**
    
* `features` : Node features
* `edge_feat` : edge features
* `state` :  edge state change labels

#### Link state change Prediction Datasets

#### Utilities

### lib.dataloading

### lib.evaluator

### lib.nn

### lib.utils
