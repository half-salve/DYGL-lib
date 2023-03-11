# RandEdgeSampler

`class lib.dataloading.RandEdgeSampler(src_list, dst_list, seed=None)`
Base : `object`
This sampler returns negatively sampled edges by random sampling from the set of nodes
（该采样器从节点集中通过随机采样，返回负采样的边）

Parmeters:

- src_list : list of source nodes in the edges of the dynamic graph
- dst_list ：list of target nodes in the edges of the dynamic graph
- seed ：random seed

---
**Example**:

(我们通过这个采样器采样出 `size`个边)

We sample `size` edges through this sampler

```
train_src = numpy.array([0,5,6,4])
train_dst = num.array([2,5,1,2])
train_rand_sampler = RandEdgeSampler(train_src, train_dst)
_ , dst_fake = train_rand_sampler.sample(size)
```
