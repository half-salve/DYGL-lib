# TGATConv

> `class lib.nn.TGATConv(ngh_finders, n_feat, e_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', 
                 num_layers=3, n_head=4, null_idx=0, drop_out=0.1, seq_len=None)`

Bases : `torch.nn.modules.module.Module`

TGNonv from [Inductive Representation Learning on Temporal Graphs](https://arxiv.org/abs/2002.07962)

**Parameters**

- ngh_finders :`[train_ngh_finder,full_ngh_finder]` You can use `lib.dataloading.NeighborFinder` and `lib.utils.init_adjacency_list` to initialize Neighbor, which is a list that needs to contain training and testing.
- n_feat : Node features
- e_feat : edge features
- num_layers ：number of network layers
- n_head ：number of heads used in tree-shaped attention layer, we only use the default here
- drop_out : Dropout probability
- num_neighbors ：a list of neighbor sampling numbers for different hops, when only a single element is input n_layer
- attn_mode ：`{prod,map}` use dot product attention or mapping based, we only use the default here
- agg_method ：`{attn,lstm,mean}` local aggregation method, we only use the default here
- use_time ：`{time,pos,empty}` how to use time information, we only use the default here

---

> `forward(node_idxs, cut_time, num_neighbors=None)`

Compute graph network layer

**Parameters**

- node_idxs : `numpy.ndarray` : Input a batch source node set
- cut_time : `numpy.ndarray` : Input a batch edge time set
- num_neighbors : `int`: The number of neighbor samples

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

src_embed = model(train_src, time )
target_embed = model(dst_cut, time )
background_embed = model(dst_fake,time )


_ , dst_fake = valid_rand_sampler.sample(size)


model.set_state("eval")

src_embed = model(src_cut, time )
target_embed = model(dst_cut, time )
background_embed = model(dst_fake,time )

```
