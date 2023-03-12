# DYGLDataset

`class lib.data.DGLDataset(name, url=None, raw_dir=None, save_dir=None, hash_key=(), force_reload=False, verbose=False, transform=None)`
Base : `Object`


The basic DYGL dataset for creating graph datasets.This class defines a basic template class for DYGL Dataset.

It is completed based on dgl's [DGLDataset](https://docs.dgl.ai/generated/dgl.data.DGLDataset.html).Although the framework remains unchanged, there are too manyfunctions that need to be rewritten, so we use transplantation instead of rewriting.

Reference : https://docs.dgl.ai/generated/dgl.data.DGLDataset.html

The following steps will be executed automatically:

1. Check whether there is a dataset cache on disk

(already processed and stored on the disk) by

invoking``has_cache()``. If true, goto 5.

2. Call``download()`` to download the data if ``url`` is not None.

3. Call``process()`` to process the data.

4. Call``save()`` to save the processed dataset on disk and goto 6.

5. Call``load()`` to load the processed dataset from disk.

6. Done.

Users can overwite these functions with their own data processing logic.

Parameters：

- name (str) – Name of the dataset

- url (str) – Url to download the raw dataset. Default: None

- raw_dir (str) – Specifying the directory that will store the downloaded data or the directory that already stores the input data. Default:`os.getwd()/DYGL_dataset/`

- save_dir (str) – Directory to save the processed dataset. Default: same as raw_dir

- hash_key (tuple) – A tuple of values as the input for the hash function. Users can distinguish instances (and their caches on the disk) from the same dataset class by comparing the hash values. Default: (), the corresponding hash value is 'f9065fa7'.

- force_reload (bool) – Whether to reload the dataset. Default: False


`url`
    The URL to download the dataset
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`name`
    The dataset name
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`raw_dir`
    Directory to store all the downloaded raw datasets.
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`raw_path`
    Path to the downloaded raw dataset folder. An alias for os.path.join(self.raw_dir, self.name).
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`save_dir`
    Directory to save all the processed datasets.
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`save_path`
    Path to the processed dataset folder. An alias for os.path.join(self.save_dir, self.name).
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`verbose`
    Whether to print more runtime information.
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`hash`
    Hash value for the dataset and the setting.
    **Type**    [str](https://docs.python.org/3/library/stdtypes.html#str)

`abstract__getitem__(idx)[source]`
    Gets the data object at index.

`abstract__len__()[source]`
    The number of examples in the dataset