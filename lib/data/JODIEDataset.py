import os

from .utils import file_download, process_edge_prediction
import numpy as np
import dgl
import torch
from tqdm import tqdm

from .dynamic_dataset  import DYGLDataset
from .init_feat import PositionalEncode
from dgl.data.graph_serialize import save_graphs,load_graphs
from dgl.data.utils import generate_mask_tensor

class  JODIEDataset(DYGLDataset):
    '''
    paper https://doi.org/10.1145/3292500.3330895 supported are
    wikipedia, reddit

    '''
    def __init__(self, name,raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0 ):
        self._name = name
        _url='https://snap.stanford.edu/jodie/{}.csv'.format(self._name)

        self.start_id = start_id
        super(JODIEDataset, self).__init__(name=name,
                                            url=_url,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform )
        

    def process(self):
        PATH = os.path.join(self.raw_path, '{}.csv'.format(self._name))

        src_ids, dst_ids, ts_list, label_list = [], [], [], []
        feat_l = []
        idx_list = []

        with open(PATH) as f:
            s = next(f)
            for idx, line in tqdm(enumerate(f)):
                e = line.strip().split(',')
                u = int(e[0])
                i = int(e[1])

                ts = int(float(e[2]))
                label = int(e[3])
                feat = np.array([float(x) for x in e[4:]])

                src_ids.append(u)
                dst_ids.append(i)
                ts_list.append(ts)
                label_list.append(label)
                idx_list.append(idx)
                feat_l.append(feat)


        time = np.array(ts_list)
        label_f = np.array(label_list)[:, np.newaxis]

        assert (max(src_ids) - min(src_ids) + 1 == len(set(src_ids))) , "The source node numbers are not consecutive."
        assert (max(dst_ids) - min(dst_ids) + 1 == len(set(dst_ids))) , "The Target node numbers are not consecutive."
        assert min(src_ids) == 0 , "the node num not start at 0"

        max_src_ids = max(src_ids) + 1
        dst_ids = [x + max_src_ids for x in dst_ids]

        #划分边集，将整体划分为0.7 0.15 0.15
        val_time, test_time = np.quantile(time, [0.70, 0.85])
        train_edge_mask = (ts_list <= val_time)
        valid_edge_mask = (ts_list <= test_time) * (ts_list > val_time)
        test_edge_mask = ts_list > test_time

        indices = time.argsort()
        feat_l = np.array(feat_l)[indices]
        if self.start_id < 0:
            raise "start_id must >=0"
        elif self.start_id > 0:
            src_ids = [x + self.start_id for x in src_ids]
            dst_ids = [x + self.start_id for x in dst_ids]
            
        self._graph = dgl.graph((src_ids,dst_ids))
        self._graph , train_edge_observed_mask,valid_edge_observed_mask,test_edge_observed_mask = process_edge_prediction(self._graph,time,train_val=0.7,val_test=0.85)


        self._graph.edata['train_edge_mask'] = generate_mask_tensor(train_edge_mask) 
        self._graph.edata['valid_edge_mask'] = generate_mask_tensor(valid_edge_mask)
        self._graph.edata['test_edge_mask'] = generate_mask_tensor(test_edge_mask)

        self._graph.edata['train_edge_observed_mask'] = generate_mask_tensor(train_edge_observed_mask)
        self._graph.edata['valid_edge_observed_mask'] = generate_mask_tensor(valid_edge_observed_mask)
        self._graph.edata['test_edge_observed_mask'] = generate_mask_tensor(test_edge_observed_mask)
        
        if self.name == "lastfm" or self.name == "mooc":
            edge_feat_init = PositionalEncode(172,self._graph.number_of_edges())
            if self.name == "mooc":
                edge_feat_init[:,:feat_l.shape[1]] = torch.tensor(feat_l)
            feat_l = edge_feat_init.numpy()
        self._graph.edata['edge_feat'] = torch.tensor(feat_l)
        self._graph.ndata["feat"] = torch.zeros((self._graph.number_of_nodes() , \
                                        self._graph.edata['edge_feat'].shape[1]))
        
        self._graph.edata['time'] = torch.tensor(time)
        self._graph.edata['state'] = torch.tensor(label_f)


        self._print_info()


    def download(self):
        PATH = os.path.join(self.raw_path, '{}.csv'.format(self._name))
        print("Start Downloading File....")
        file_download(self.url,PATH)


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

class RedditDataset(JODIEDataset):

    '''

    The authors selected the 1,000
    most active subreddits as items and the 10,000 most active users.
    This results in 672,447 interactions. The authors convert the text of each
    post into a feature vector representing their LIWC categories.

    We do the chronological train-validation-test split with 70%-15%-15% according to node interaction
    timestamps. The dataset and preprocessing details are provided in the supplement material.

    Reference:
     @inproceedings{kumar2019predicting,
	title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
	author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
	booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
	year={2019},
	organization={ACM}
    }
    
    Statistics

    - Nodes: 11,000
    - Edges: 672,447
    - Node feature size: 172
    - Nodes with dynamic labels
    - Feature type: LIWC category vector
    
    Attributes
    ----------

    graph : :class:`dgl.DGLGraph`
        Graph of the dataset
    time: Timestamps on edges sorted by time
    train_edge_mask: numpy.ndarray 
        positive training edge mask for **Transductive task**
    valid_edge_mask: numpy.ndarray
        positive validation edge mask for **Transductive task**
    test_edge_mask: numpy.ndarray 
        positive testing edge mask for **Transductive task**
    train_edge_observed_mask: numpy.ndarray
        positive training edge mask for **Inductive task**
    valid_edge_observed_mask: numpy.ndarray
        positive validation edge mask for **Inductive task**
    test_edge_observed_mask: numpy.ndarray
        positive testing edge mask for **Inductive task**
    features : Node features
    edge_feat : edge features
    state :  edge state change labels


    '''
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0):
        name="reddit"
        super(RedditDataset, self).__init__(name=name,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform ,start_id=start_id)


    def __len__(self):
        r"""Number of graphs in the dataset"""
        return super(JODIEDataset,self).__len__()


class WikipediaDataset(JODIEDataset):

    '''
    

    The authors selected the 1,000 most edited
    pages as items and editors who made at least 5 edits as users (a total
    of 8,227 users). This generates 157,474 interactions. Similar to the
    Reddit dataset, The authors convert the edit text into a LIWC-feature vector

    We do the chronological train-validation-test split with 70%-15%-15% according to node interaction
    timestamps. The dataset and preprocessing details are provided in the supplement material.

    Reference:
    @inproceedings{kumar2019predicting,
    title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
    author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
    booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
    year={2019},
    organization={ACM}
    }
    
    Statistics

    - Nodes: 9,227
    - Edges: 157,474
    - Node feature size: 172
    - Nodes with dynamic labels
    - Feature type: LIWC category vector
    
    Attributes
    ----------
    
    graph : :class:`dgl.DGLGraph`
        Graph of the dataset
    time: Timestamps on edges sorted by time
    train_edge_mask: numpy.ndarray 
        positive training edge mask for **Transductive task**
    valid_edge_mask: numpy.ndarray
        positive validation edge mask for **Transductive task**
    test_edge_mask: numpy.ndarray 
        positive testing edge mask for **Transductive task**
    train_edge_observed_mask: numpy.ndarray
        positive training edge mask for **Inductive task**
    valid_edge_observed_mask: numpy.ndarray
        positive validation edge mask for **Inductive task**
    test_edge_observed_mask: numpy.ndarray
        positive testing edge mask for **Inductive task**
    features : Node features
    edge_feat : edge features
    state :  edge state change labels


    '''
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0):
        name="wikipedia"
        super(WikipediaDataset, self).__init__(name=name,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform ,start_id=start_id )

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return super(JODIEDataset,self).__len__()
    
class MOOCDataset(JODIEDataset):

    '''
    

    this public dataset consists of actions,
    e.g., viewing a video, submitting an answer, etc., done by students
    on a MOOC online course . This dataset consists of 7,047 users
    interacting with 98 items (videos, answers, etc.) resulting in over
    411,749 interactions. There are 4,066 drop-out events (= 0.98%).

    We do the chronological train-validation-test split with 70%-15%-15% according to node interaction
    timestamps. The dataset and preprocessing details are provided in the supplement material.

    Reference:
    @inproceedings{kumar2019predicting,
    title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
    author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
    booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
    year={2019},
    organization={ACM}
    }
    
    Statistics

    - Nodes: 7,144
    - Edges: 411,749 
    - Node feature size: 4
    - Nodes with dynamic labels
    - Feature type: LIWC category vector
    
    Attributes
    ----------

    graph : :class:`dgl.DGLGraph`
        Graph of the dataset

    time: Timestamps on edges sorted by time
    train_edge_mask: numpy.ndarray 
        positive training edge mask for **Transductive task**
    valid_edge_mask: numpy.ndarray
        positive validation edge mask for **Transductive task**
    test_edge_mask: numpy.ndarray 
        positive testing edge mask for **Transductive task**
    train_edge_observed_mask: numpy.ndarray
        positive training edge mask for **Inductive task**
    valid_edge_observed_mask: numpy.ndarray
        positive validation edge mask for **Inductive task**
    test_edge_observed_mask: numpy.ndarray
        positive testing edge mask for **Inductive task**
    features : Node features
    edge_feat : edge features
    state :  edge state change labels


    '''
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0):
        name="mooc"
        super(MOOCDataset, self).__init__(name=name,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform ,start_id=start_id)

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return super(JODIEDataset,self).__len__()
    
class LastFMDataset(JODIEDataset):

    '''
     this public dataset has one month of who-listens-to-which song information . The authors selected all 1000 users 
     and the 1000 most listened songs resulting in 1,293,103 interactions.
     In this dataset, interactions do not have features.

    We do the chronological train-validation-test split with 70%-15%-15% according to node interaction
    timestamps. The dataset and preprocessing details are provided in the supplement material.

    Reference:
    @inproceedings{kumar2019predicting,
    title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
    author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
    booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
    year={2019},
    organization={ACM}
    }
    
    Statistics

    - Nodes: 1980
    - Edges: 1,293,103
    - Node feature size: None
    - Nodes with dynamic labels
    
    Attributes
    ----------

    graph : :class:`dgl.DGLGraph`
        Graph of the dataset

    time: Timestamps on edges sorted by time
    train_edge_mask: numpy.ndarray 
        positive training edge mask for **Transductive task**
    valid_edge_mask: numpy.ndarray
        positive validation edge mask for **Transductive task**
    test_edge_mask: numpy.ndarray 
        positive testing edge mask for **Transductive task**
    train_edge_observed_mask: numpy.ndarray
        positive training edge mask for **Inductive task**
    valid_edge_observed_mask: numpy.ndarray
        positive validation edge mask for **Inductive task**
    test_edge_observed_mask: numpy.ndarray
        positive testing edge mask for **Inductive task**
    features : Node features
    edge_feat : edge features
    state :  edge state change labels

    '''
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None,start_id=0):
        name="lastfm"
        super(LastFMDataset, self).__init__(name=name,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform ,start_id=start_id)

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return super(JODIEDataset,self).__len__()