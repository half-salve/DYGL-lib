"""Dataset utilities."""
import os
import errno
import requests
import random
from tqdm import tqdm

import numpy as np
import wget
import dgl

name="DYGL_dataset"

def get_download_dir():
    """Get the absolute path to the download directory.

    Returns
    -------
    dirname : str
        Path to the download directory
    """
    default_dir = os.path.join(os.getcwd(), name)#name="DYGL_dataset"
    dirname = os.environ.get('DYGL_DOWNLOAD_DIR', default_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def makedirs(path):
    try:
        os.makedirs(os.path.normpath(path))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def file_download(url: str, fname: str):
    # 用流stream的方式获取url的数据
    resp = requests.get(url, stream=True)
    # 拿到文件的长度，并把total初始化为0
    total = int(resp.headers.get('content-length', 0))
    # 打开当前目录的fname文件(名字你来传入)
    # 初始化tqdm，传入总数，文件名等数据，接着就是写入，更新等操作了
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def wget_download(url: str, fname: str):
    wget.download(url,fname)


def process_edge_prediction(graph,time,train_val,val_test):
    src ,dst = graph.edges()

    src ,dst= np.array(src), np.array(dst)

    indices = time.argsort()

    sources ,destinations ,time = src[indices], dst[indices], time[indices]

    val_time, test_time = np.quantile(time, [train_val, val_test])

    node_set = set(list(sources)) | set(list(destinations))
    n_total_unique_nodes = len(node_set)
    

    test_node_set = set(sources[time > val_time]).union(
                        set(destinations[time > val_time]))
    
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = np.array([x in new_test_node_set for x in sources])
    new_test_destination_mask = np.array([x in new_test_node_set for x in destinations])

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    #train_data中无选中的节点
    train_edge_observed_mask = np.logical_and(time <= val_time, observed_edges_mask)

    #无选中节点的集合
    train_node_set = set(sources[train_edge_observed_mask]).union(destinations[train_edge_observed_mask])
    assert len(train_node_set & new_test_node_set) == 0

    #train_data中不存在节点的集合
    new_node_set = node_set - train_node_set

    #在时间限定范围内的边
    val_mask = (time <= test_time) * (time> val_time)
    test_mask= time > test_time

    #在valid、test中 有train中没有的节点的边，范围非常的小
    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    valid_edge_observed_mask = val_mask * is_new_node_edge
    test_edge_observed_mask = test_mask * is_new_node_edge

    g = dgl.graph((sources,destinations))
    
    return g ,train_edge_observed_mask , valid_edge_observed_mask ,test_edge_observed_mask

