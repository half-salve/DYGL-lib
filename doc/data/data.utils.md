# get_download_dir

`class lib.data.utils.get_download_dir`
    Get the absolute path to the download directory.
        **Returns**             **dirname** - Path to the download directory
        **Return type**         [str](https://docs.python.org/3/library/stdtypes.html#str)

# file_download
`class lib.data.utils.file_download(url: str, fname: str)`
    Download a given URL
    **Parmeters**
    - **url** : URL to download
    - **fname** : Destination path to store downloaded file. By default stores to the current directory with the same name as in url.

# process_edge_prediction
`class lib.data.utils.process_edge_prediction(graph,time,train_val,val_test)`
After sorting the edges of the dynamic graph according to the values in `time`, divide the first `train_va`l edges of the input dynamic graph table into `train_edge_observed_mask`, `train_val`-`val_test` edges into `valid_edge_observed_mask`, and the part after `val_test` into `test_edge_observed_mask`
    **Parmeters**
- **graph** : URL to download
- **time** : Destination path to store downloaded file. By default stores to the current directory with the same name as in url.
- **train_val** : The percentage of the train dataset to the total dataset(0<train_val<1)
- **val_test** : The percentage of train and valid dataset to the total dataset(0<val_test<1)

    **Returns**             
- **g** - The dynamic graph reconstructed after sorting the edges of the dynamic graph according to the value in time
    **Return type**         DGLGraph
- **train_edge_observed_mask** - The edge of g is divided into the mask of train
    **Return type**         torch.Tensor
- **valid_edge_observed_mask** - The edge of g is divided into the mask of valid
    **Return type**         torch.Tensor
- **test_edge_observed_mask** - The edge of g is divided into the mask of test
    **Return type**         torch.Tensor