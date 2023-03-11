# CAWConv

> `class lib.nn.CAWConv(ngh_finders,n_feat, e_feat, agg='tree',attn_mode='prod', use_time='time', attn_agg_method='attn', pos_dim=0, pos_enc='spd', walk_pool='attn', walk_n_head=8, walk_mutual=False, num_layers=3, n_head=4, drop_out=0.1, num_neighbors=20, cpu_cores=1,verbosity=1, walk_linear_out=False)`

Bases : `torch.nn.modules.module.Module`

CAWConv from [INDUCTIVE REPRESENTATION LEARNING IN TEMPORAL NETWORKS VIA CAUSAL ANONYMOUS WALKS](https://arxiv.org/pdf/2101.05974.pdf)

**Parameters**

- ngh_finders :`[train_ngh_finder,full_ngh_finder]` You can use `lib.dataloading.NeighborFinder` and `lib.utils.init_adjacency_list` to initialize Neighbor, which is a list that needs to contain training and testing.
- n_feat : Node features
- e_feat : edge features
- num_layers ：number of network layers
- n_head ：number of heads used in tree-shaped attention layer, we only use the default here
- drop_out
- num_neighbors ：a list of neighbor sampling numbers for different hops, when only a single element is input n_layer
- agg ：`{tree,walk}` tree based hierarchical aggregation or walk-based flat lstm aggregation
- attn_mode ：`{prod,map}` use dot product attention or mapping based, we only use the default here
- use_time ：`{time,pos,empty}` how to use time information, we only use the default here
- attn_agg_method ：`{attn,lstm,mean}` local aggregation method, we only use the default here
- pos_dim ：dimension of the positional embedding
- pos_enc ：`{spd,lp,saw}` way to encode distances, shortest-path distance or landing probabilities, or self-based anonymous
- walk_pool ： `{attn,sum}` how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other
- walk_n_head ：number of heads to use for walk attention
- walk_mutual ：whether to do mutual query for source and target node random walks
- cpu_cores ：number of cpu_cores used for position encoding
- verbosity ：verbosity of the program output
- walk_linear_out ：whether to linearly project each node's

---

> `forward(src_idx, tgt_idx , cut_time, e_idx=None, flag_for_cur_edge=True)`

Compute graph network layer
**Parameters**

- src_idx : `numpy.ndarray` : Input a batch source node set
- tgt_idx : `numpy.ndarray` ：Input a batch target node set
- cut_time : `numpy.ndarray` : Input a batch edge time set
- e_idx : `numpy.ndarray` : Input a batch edge index set
- flag_for_cur_edge : `bool`: If the input is a positive example of an edge, it is set to True, and a negative example is set to False

> `set_state(state)`

**Parameters**

- state : `model.set_state("train")` is equivalent to `model.train()` , `model.set_state("eval")` is equivalent to `model.eval()`

---

**Example**:

```
train_src = numpy.array([0,5,6,4])
train_dst = num.array([2,5,1,2])

valid_src = numpy.array([4,8,6,1])
valid_dst = num.array([6,5,0,2])

train_rand_sampler = RandEdgeSampler(train_src, train_dst)
valid_rand_sampler = RandEdgeSampler(valid_src, valid_dst)

edge_cut = numpy.arange(len(train_src))

size = train_src.shape[0]

model.set_state("train")
_ , dst_fake = train_rand_sampler.sample(size)
src_embed , target_embed= model(src_cut, dst_cut ,time_cut , edge_cut ,True)
_ , background_embed = model(src_cut , dst_fake, time_cut ,edge_cut , False)

model.set_state("eval")
_ , dst_fake = valid_rand_sampler.sample(size)
src_embed , target_embed= model(src_cut, dst_cut ,time_cut , edge_cut ,True)
_ , background_embed = model(src_cut , dst_fake, time_cut ,edge_cut , False)

```
