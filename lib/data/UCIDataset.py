import os
import sys
import datetime

from .utils import wget_download, process_edge_prediction
import numpy as np
import pandas as  pd

import dgl
import torch
from tqdm import tqdm

from .dynamic_dataset  import DYGLDataset

from dgl.data.graph_serialize import save_graphs,load_graphs
from dgl.data.utils import generate_mask_tensor

class  UCIDataset(DYGLDataset):
    '''

    '''
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0 ):
        name = "UCI"
        _url='http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt'

        self.start_id = start_id
        super(UCIDataset, self).__init__(name=name ,
                                            url=_url,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform )
        

    def process(self):
        PATH = os.path.join(self.raw_path, '{}.txt'.format(self._name))
        data  = pd.read_table(PATH ,sep=" ",header=None)
        data.columns=["time","source","target","label"]
        print(data["time"][0])
        format = "%Y-%m-%d %H:%M:%S"
        start_time = datetime.datetime.strptime(data["time"][0],format)
        time = [(datetime.datetime.strptime(t,format) - start_time).seconds for t in data["time"]]
        second_time = np.array(time)
        src_ids = data["source"].values
        dst_ids = data["target"].values
        indexs = []
        for index, (source , target) in enumerate(zip(src_ids,dst_ids)) :
            if source == target:
                indexs.append(index)

        second_time = np.delete(second_time,indexs)
        src_ids = np.delete(src_ids,indexs)
        dst_ids = np.delete(dst_ids,indexs)

        min_id = min(src_ids.min(),dst_ids.min())
        max_id = max(src_ids.max(),dst_ids.max())
        
        assert (max_id - min_id + 1 == len(np.unique(np.union1d(src_ids ,dst_ids))) ), "The source node numbers are not consecutive."

        dst_ids = dst_ids - min_id
        src_ids = src_ids - min_id
        #划分边集，将整体划分为0.7 0.15 0.15
        val_time, test_time = np.quantile(second_time, [0.70, 0.85])
        train_edge_mask = (second_time <= val_time)
        valid_edge_mask = (second_time <= test_time) * (second_time > val_time)
        test_edge_mask = second_time > test_time

        if self.start_id < 0:
            raise "start_id must >=0"
        elif self.start_id > 0:
            src_ids = [x + self.start_id for x in src_ids]
            dst_ids = [x + self.start_id for x in dst_ids]
            
        self._graph = dgl.graph((src_ids,dst_ids))
        self._graph , train_edge_observed_mask,valid_edge_observed_mask,test_edge_observed_mask = process_edge_prediction(self._graph,second_time,train_val=0.7,val_test=0.85)


        self._graph.edata['train_edge_mask'] = generate_mask_tensor(train_edge_mask) 
        self._graph.edata['valid_edge_mask'] = generate_mask_tensor(valid_edge_mask)
        self._graph.edata['test_edge_mask'] = generate_mask_tensor(test_edge_mask)

        self._graph.edata['train_edge_observed_mask'] = generate_mask_tensor(train_edge_observed_mask)
        self._graph.edata['valid_edge_observed_mask'] = generate_mask_tensor(valid_edge_observed_mask)
        self._graph.edata['test_edge_observed_mask'] = generate_mask_tensor(test_edge_observed_mask)
        
        self._graph.edata['edge_feat'] = torch.zeros((self._graph.number_of_edges(),64))

        self._graph.ndata["feat"] = torch.zeros((self._graph.number_of_nodes() ,64))
        self._graph.edata['time'] = torch.tensor(second_time/3600)
        print("processed raw data")
        self._print_info()


    def download(self):
        PATH = os.path.join(self.raw_path, '{}.txt'.format(self._name))
        print("Start Downloading File....")
        wget_download(self.url,PATH)
        print("finished Downloading File....")


    def has_cache(self):

        graph_path = os.path.join(self.save_path, 'dgl_graph_{}.bin'.format(self.start_id))
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_{}.bin'.format(self.start_id))
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_{}.bin'.format(self.start_id))
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]

        self._graph.edata['train_edge_mask'] = generate_mask_tensor(self._graph.edata['train_edge_mask'].numpy())
        self._graph.edata['valid_edge_mask'] = generate_mask_tensor(self._graph.edata['valid_edge_mask'].numpy())
        self._graph.edata['test_edge_mask'] = generate_mask_tensor(self._graph.edata['test_edge_mask'].numpy())

        self._graph.edata['train_edge_observed_mask'] = generate_mask_tensor(self._graph.edata['train_edge_observed_mask'].numpy())
        self._graph.edata['valid_edge_observed_mask'] = generate_mask_tensor(self._graph.edata['valid_edge_observed_mask'].numpy())
        self._graph.edata['test_edge_observed_mask'] = generate_mask_tensor(self._graph.edata['test_edge_observed_mask'].numpy())
        self._print_info()

    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumNodeFeats: {}'.format(self._graph.ndata['feat'].shape))
            print('  NumEdgeFeats: {}'.format(self._graph.edata['edge_feat'].shape))

            print('  train_edge_observed_mask: {}/{}'.format( (self._graph.edata['train_edge_observed_mask']==True).sum(), self._graph.edata['train_edge_observed_mask'].shape[0]))
            print('  valid_edge_observed_mask: {}/{}'.format( (self._graph.edata['valid_edge_observed_mask']==True).sum(), self._graph.edata['valid_edge_observed_mask'].shape[0]))
            print('  test_edge_observed_mask: {}/{}'.format( (self._graph.edata['test_edge_observed_mask']==True).sum(), self._graph.edata['test_edge_observed_mask'].shape[0]))

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)