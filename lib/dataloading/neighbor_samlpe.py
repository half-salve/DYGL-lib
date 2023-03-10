import numpy as np
import torch
from tqdm import tqdm
'''
@article{DBLP:journals/corr/abs-2002-07962,
  author    = {Da Xu and
               Chuanwei Ruan and
               Evren K{\"{o}}rpeoglu and
               Sushant Kumar and
               Kannan Achan},
  title     = {Inductive Representation Learning on Temporal Graphs},
  journal   = {CoRR},
  volume    = {abs/2002.07962},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.07962},
  eprinttype = {arXiv},
  eprint    = {2002.07962},
  timestamp = {Mon, 02 Mar 2020 16:46:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2002-07962.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}

'''


class TGATNeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: Tensor[int]
        node_ts_l: Tensor[int]
        off_set_l: Tensor[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
        
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        
        for i in tqdm(range(len(adj_list))):#遍历所有节点
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])

            #按序号的顺序排序
            n_idx_l.extend([x[0].view(-1) for x in curr])
            e_idx_l.extend([x[1].view(-1) for x in curr])
            n_ts_l.extend([x[2].view(-1) for x in curr])
            off_set_l.append(len(n_idx_l))
        '''
            用off_set_l记录，每个节点邻居的个数，
            将adj_list中所有的信息都装进了几个列表，
            无法分辨谁是谁的，猜测off_set_l起分割作用
        '''
        # print(n_idx_l)
        n_idx_l = torch.cat(n_idx_l)
        n_ts_l = torch.cat(n_ts_l)
        e_idx_l = torch.cat(e_idx_l)
        off_set_l = torch.tensor(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l,  e_idx_l , off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        #off_set_l[src_idx]：目标节点的邻居开始的下标.

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        #提取出目标节点的所有邻居（可重复）
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        
        if neighbors_idx.shape[0] == 0 or neighbors_ts.shape[0] == 0:
            return neighbors_idx, neighbors_ts,neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
                
        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]
        #返回给出的时间坐标前的邻居

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert((src_idx_l.shape[0]) == (cut_time_l).shape[0])
        
        out_ngh_node_batch = torch.zeros((src_idx_l.shape[0], num_neighbors),dtype=torch.int32)
        out_ngh_t_batch = torch.zeros((src_idx_l.shape[0], num_neighbors),dtype=torch.float32)
        out_ngh_eidx_batch = torch.zeros((src_idx_l.shape[0], num_neighbors),dtype=torch.int32)
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            #取出一个batch_size，也就是边里的每一个源节点和边对应的形成时间
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if ngh_idx.shape[0] > 0:
                if self.uniform:
                    sampled_idx = torch.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                    print("YRUE")
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    
                    assert(ngh_idx.shape[0] <= num_neighbors)
                    assert(ngh_ts.shape[0] <= num_neighbors)
                    assert(ngh_eidx.shape[0] <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - ngh_idx.shape[0]:] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - ngh_ts.shape[0]:] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - ngh_eidx.shape[0]:] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors, e_idx_l=None):
        """Sampling the k-hop sub graph in tree struture
        """
        if k == 0:
            return ([], [], [])
        batch = len(src_idx_l)
        layer_i = 0
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors[layer_i], e_idx_l=e_idx_l)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for layer_i in range(1, k):
            ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_records[-1]
            ngh_node_est = ngh_node_est.flatten()
            ngh_e_est = ngh_e_est.flatten()
            ngh_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est, ngh_t_est, num_neighbors[layer_i], e_idx_l=ngh_e_est)
            out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1)
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(batch, -1)
            out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)

        return (node_records, eidx_records, t_records)  # each of them is a list of k numpy arrays, each in shape (batch,  num_neighbors ** hop_variable)

class NeighborFinder:
    def __init__(self,adj_list, uniform=False, seed=None):

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)
        self.uniform = uniform

        self.node_to_neighbors,self.node_to_edge_idxs ,self.node_to_edge_timestamps= self.init_off_set(adj_list)

    def init_off_set(self, adj_list):
        node_to_neighbors = []
        node_to_edge_idxs = []
        node_to_edge_timestamps = []
        adj_ = adj_list
        for neighbors in adj_:
        # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
        # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        return node_to_neighbors,node_to_edge_idxs,node_to_edge_timestamps

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                        timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

        if len(source_neighbors) > 0 and n_neighbors > 0:
            if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                neighbors[i, :] = source_neighbors[sampled_idx]
                edge_times[i, :] = source_edge_times[sampled_idx]
                edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                # re-sort based on time
                pos = edge_times[i, :].argsort()
                neighbors[i, :] = neighbors[i, :][pos]
                edge_times[i, :] = edge_times[i, :][pos]
                edge_idxs[i, :] = edge_idxs[i, :][pos]
            else:
            # Take most recent interactions
                source_edge_times = source_edge_times[-n_neighbors:]
                source_neighbors = source_neighbors[-n_neighbors:]
                source_edge_idxs = source_edge_idxs[-n_neighbors:]

                assert (len(source_neighbors) <= n_neighbors)
                assert (len(source_edge_times) <= n_neighbors)
                assert (len(source_edge_idxs) <= n_neighbors)

                neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times
    
