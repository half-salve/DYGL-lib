# DyRepConv

> `class lib.nn.DyRepConv(ngh_finders, n_feat, e_feat, device, n_layers=2, n_heads=2, dropout=0.1,  use_memory=True, memory_update_at_start=True,  message_dimension=100, memory_dimension=500,  embedding_module_type="graph_attention", message_function="identity",  mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0, std_time_shift_dst=1,  n_neighbors=20, aggregator_type="last", memory_updater_type="rnn",  use_destination_embedding_in_message=True, use_source_embedding_in_message=False,  dyrep=True)`

Bases : `torch.nn.modules.module.Module`

TGNonv from [Learning Representation over Dynamic Graph](https://openreview.net/pdf?id=HyePrhR5KX)

**Parameters**

- ngh_finders :`[train_ngh_finder,full_ngh_finder]` You can use `lib.dataloading.NeighborFinder` and `lib.utils.init_adjacency_list` to initialize Neighbor, which is a list that needs to contain training and testing.
- n_feat : Node features
- e_feat : edge features
- device : Whether the device is running on cpu or gpu
- n_layers ：number of network layers
- n_head ：number of heads used in tree-shaped attention layer, we only use the default here
- dropout : Dropout probability
- num_neighbors ：a list of neighbor sampling numbers for different hops, when only a single element is input n_layer
- use_memory ：Whether to use a memory for the nodes
- memory_update_at_start ：Whether to update the memory at the end or at the start of the batch
- message_dimension ：Dimension of the messages
- memory_dimension ：Dimension of the memory
- embedding_module_type ：Type of the embedding module
- message_function ：Type of the message functionprobabilities, or self-based anonymous
- aggregator_type ： Type of the message aggregator
- memory_updater_type ：Type of the memory updater
- dyrep : Whether to run the model as DyRep

---

> `forward(source_nodes, destination_nodes, negative_nodes = None, edge_times=None, edge_idxs=None, n_neighbors=None)`

Compute graph network layer

**Parameters**

- source_nodes : `numpy.ndarray` : Input a batch source node set
- destination_nodes : `numpy.ndarray` ：Input a batch target node set
- negative_nodes : `numpy.ndarray` : Input a batch edge time set. Can be entered as None
- edge_times : `numpy.ndarray` : Input a batch edge time set
- edge_index : `numpy.ndarray` : Input a batch edge index set
- n_neighbors : `int`: The number of neighbor samples

> `set_state(state)`

**Parameters**

- state : `model.set_state("train")` is equivalent to `model.train()` , `model.set_state("eval")` is equivalent to `model.eval()`

---

**Example**:

```python
train_src = numpy.array([0,5,6,4])
train_dst = num.array([2,5,1,2])

valid_src = numpy.array([4,8,6,1])
valid_dst = num.array([6,5,0,2])

edge_cut = numpy.arange(len(train_src))
time = numpy.random.randint(0,10,train.shape)

train_rand_sampler = RandEdgeSampler(train_src, train_dst)
valid_rand_sampler = RandEdgeSampler(valid_src, valid_dst)

size = train_src.shape[0]

_ , dst_fake = train_rand_sampler.sample(size)

model.set_state("train")
src_embed, target_embed,background_embed= model(source_nodes=src_cut,destination_nodes = dst_cut,
                            negative_nodes=dst_fake, edge_times = time_cut, edge_idxs = edge_cut)

_ , dst_fake = valid_rand_sampler.sample(size)

model.set_state("eval")
src_embed, target_embed,background_embed= model(source_nodes=train_src,destination_nodes = valid_dst,
                            negative_nodes=dst_fake, edge_times = time_cut, edge_idxs = edge_cut)
```
