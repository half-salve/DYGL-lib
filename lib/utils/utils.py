import logging
import datetime
import os
import sys
import numpy as np
import random
import torch
from ..data import RedditDataset,WikipediaDataset,MOOCDataset,LastFMDataset,UCIDataset

def get_logger(parms,log_level='INFO',save_dir=None ,name=None,):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    if save_dir is None:
        log_dir = os.path.join(os.getcwd(),"log")
    else:
        log_dir = save_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = "-".join(parms) + ".log"
    
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)


    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger

def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_checkpoints_metric(config,parms):
    parms = [str(i) for i in parms]
    checkpoints_root = './saved_checkpoints/'
    if not os.path.exists(checkpoints_root):
        os.makedirs(checkpoints_root)
        print('Create directory {}'.format(checkpoints_root))

    path = "-".join(parms)
    saved_checkpoints_dir = f'{checkpoints_root}/{config["model"]}/{config["dataset"]}/{config["task"]}/{path}/'
    if not os.path.exists(saved_checkpoints_dir):
        os.makedirs(saved_checkpoints_dir)
        print('Create directory {}'.format(saved_checkpoints_dir))
    
    if not os.path.exists(saved_checkpoints_dir):
        os.makedirs(saved_checkpoints_dir)

    checkpoints_path = lambda epoch: f'{saved_checkpoints_dir}/{config["prefix"]}-epoch-{epoch}.pth'

    save_dir= os.path.join("./metric",config["model"],config["dataset"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return checkpoints_path,save_dir

def data_from_mask(graph,mask,start_id=0):

    src ,dst = graph.edges()[0] ,graph.edges()[1]
    t = graph.edata["time"]
    edge_idx = torch.arange(start_id,graph.number_of_edges()+start_id)

    sources_mask =  src[mask]

    Targets_mask = dst[mask]
    
    time_mask = t[mask]
    
    edges_mask = edge_idx[mask]
    
    #现在的情况是 ['train_edge_observed_mask']是[,1],[Data.edata['valid_edge_mask']]是-1，但是结果刚好相反，train中的是[-1]
    return sources_mask , Targets_mask , time_mask ,edges_mask
    
def numpy_from_mask(graph,mask,start_id=0):

    src ,dst = graph.edges()[0] ,graph.edges()[1]
    t = graph.edata["time"]
    edge_idx = torch.arange(start_id,graph.number_of_edges()+start_id)

    sources_mask =  src[mask]

    Targets_mask = dst[mask]
    
    time_mask = t[mask]
    
    edges_mask = edge_idx[mask]
    
    #现在的情况是 ['train_edge_observed_mask']是[,1],[Data.edata['valid_edge_mask']]是-1，但是结果刚好相反，train中的是[-1]
    return sources_mask.numpy() , Targets_mask.numpy() , time_mask.numpy() ,edges_mask.numpy()


def init_adjacency_list( max_idx,src,dst,edges, time):
    
    src_max = src.max()
    dst_max = dst.max()
    if max_idx <src_max or max_idx<dst_max:
        raise "max_idx is less than the maximum value of the output source or target node set"

    adj_list = [[] for _ in range(max_idx + 1)]
    for source, target, edge ,ts in zip(src,dst,edges, time):
        adj_list[source].append((target,edge, ts))
        adj_list[target].append((source, edge, ts))

    return adj_list

def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

def get_data(name, model_name, start_id=0):
    data = 0
    if model_name == "CAW":
        start_id = 1
    if name == "wikipedia":
        data = WikipediaDataset(start_id=start_id)[0]
    if name == "Reddit":
        data =  RedditDataset(start_id=start_id)[0]
    if name == "Mooc":
        data = MOOCDataset(start_id=start_id)[0]
    if name == "LastFM":
        data = LastFMDataset(start_id=start_id)[0]
    if name == "UCI":
        data = UCIDataset(start_id=start_id)[0]
    if data ==0:
        raise "no dataset"
    return data

