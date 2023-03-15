# LastFMDataset

`class lib.data.LastFMDataset(raw_dir=None, force_reload=False, verbose=True, transform=None,start_id = 0)`

Base : `lib.data.JODIEDataset`

Reddit Dataset Statistics

* Nodes: 1980
* Edges: 1,293,103
* Node feature size: 2
* Nodes with dynamic labels

Parameters:

* raw_dir (str) – Raw file directory to download/contains the input data directory. Default: ~/.dgl/
* force_reload (bool) – Whether to reload the dataset. Default: False

- verbose (bool) – Whether to print out progress information. Default: True.
- transform (callable, optional) – A transform that takes in a `DGLGraph` object and returns a transformed version. The `DGLGraph` object will be transformed before every access.
  `__getiem__(idx)`

parms ：**idx**( *int* )

Return ：This dynamic graph contains:

* `edata['time']`: Timestamps on edges sorted by time
* `edata['train_edge_mask']`: positive training edge mask for **Transductive task**
* `edata['valid_edge_mask']`: positive validation edge mask for **Transductive task**
* `edata['test_edge_mask']`: positive testing edge mask for **Transductive task**
* `edata['train_edge_observed_mask']`: positive training edge mask for **Inductive task**
* `edata['valid_edge_observed_mask']`: positive validation edge mask for **Inductive task**
* `edata['test_edge_observed_mask']`: positive testing edge mask for **Inductive task**
* `features` : Node features
* `edge_feat` : edge features
  *` state` :  edge state change labels

Return type
    `dgl.DGLGraph`
