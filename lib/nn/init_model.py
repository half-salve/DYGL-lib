from .tgatconv import TGATConv
from ..dataloading import TGATNeighborFinder,NeighborFinder

from .cawconv import CAWConv
from .cawconv import CAWNeighborFinder

from .tgnconv import TGNConv

from ..utils import numpy_from_mask,init_adjacency_list,compute_time_statistics
import torch
import sys
import numpy as np

def TGAT_init(config,Data,task):
    src ,dst = Data.edges()[0].numpy(),Data.edges()[1].numpy()
    t = Data.edata["time"].numpy()
    edge_idx = torch.arange(Data.number_of_edges()).numpy()
    ##转化为numpy格式
    max_idx = max(src.max(),dst.max())
    if task == "node_classification":
        train_src,train_dst,train_t,train_edge = numpy_from_mask(Data,Data.edata['train_edge_mask'])
    elif task == "link_prediction":
        train_src,train_dst,train_t,train_edge = numpy_from_mask(Data,Data.edata['train_edge_observed_mask'])

    else:
        raise"no task "
    #先设置好了list共有节点数个list
    print("Generate a set of neighbors for a train node>>>>>>>>")
    train_adj_list = init_adjacency_list(max_idx, train_src, train_dst, train_edge ,train_t)
    train_ngh_finder = NeighborFinder(train_adj_list, config["uniform"])
    
    print("Generate a set of neighbors for a full node>>>>>>>>")
    full_adj_list = init_adjacency_list(max_idx, src, dst, edge_idx ,t)
    full_ngh_finder = NeighborFinder(full_adj_list, config["uniform"])

    model = TGATConv([train_ngh_finder,full_ngh_finder], Data.ndata["feat"], Data.edata["edge_feat"],
            num_layers=config["n_layer"], use_time=config["time"], agg_method=config["agg_method"], attn_mode=config["attn_mode"],
            seq_len=config["n_degree"] , n_head=config["n_head"], drop_out=config["drop_out"])
    
    return model



def TGN_init(config,Data,task):
    src ,dst = Data.edges()[0].numpy(),Data.edges()[1].numpy()
    t = Data.edata["time"].numpy()
    edge_idx = torch.arange(Data.number_of_edges()).numpy()
    ##转化为numpy格式
    max_idx = max(src.max(),dst.max())

    if task == "node_classification":
        train_src,train_dst,train_t,train_edge = numpy_from_mask(Data,Data.edata['train_edge_mask'])

    elif task == "link_prediction":
        train_src,train_dst,train_t,train_edge = numpy_from_mask(Data,Data.edata['train_edge_observed_mask'])

    else:
        raise"no task "
    #先设置好了list共有节点数个list
    # Initialize training neighbor finder validation and test neighbor to retrieve temporal graph and model
    
    print("Generate a set of neighbors for a train node>>>>>>>>")
    train_adj_list = init_adjacency_list(max_idx, train_src, train_dst, train_edge ,train_t)
    train_ngh_finder = NeighborFinder(train_adj_list,config["uniform"])

    print("Generate a set of neighbors for a full node>>>>>>>>")
    full_adj_list = init_adjacency_list(max_idx, src, dst, edge_idx ,t)
    full_ngh_finder = NeighborFinder(full_adj_list,config["uniform"])
    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
                    compute_time_statistics(src, dst,  t)
    
    model = TGNConv(ngh_finders = [train_ngh_finder,full_ngh_finder], n_feat = Data.ndata["feat"],
                        e_feat= Data.edata["edge_feat"], device=config["device"],
                        n_layers = config["n_layer"],
                        n_heads = config["n_head"], dropout = config["drop_out"], use_memory = config["use_memory"],
                        message_dimension = config["message_dim"], memory_dimension = config["memory_dim"],
                        memory_update_at_start = not config["memory_update_at_end"],
                        embedding_module_type = config["embedding_module"],
                        message_function = config["message_function"],
                        aggregator_type = config["aggregator"],
                        memory_updater_type = config["memory_updater"],
                        n_neighbors = config["n_degree"],
                        mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                        mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                        use_destination_embedding_in_message=config["use_destination_embedding_in_message"],
                        use_source_embedding_in_message=config["use_source_embedding_in_message"],
                        dyrep=config["dyrep"])
    return model

def CAW_init(config,Data,task):
    src ,dst = Data.edges()[0].numpy(),Data.edges()[1].numpy()
    t = Data.edata["time"].numpy()
    edge_idx = torch.arange(1,Data.number_of_edges()+1).numpy()

    if task == "node_classification":
        train_src,train_dst,train_t,train_edge = numpy_from_mask(Data,Data.edata['train_edge_mask'],1)
            
    elif task == "link_prediction":
        train_src,train_dst,train_t,train_edge = numpy_from_mask(Data,Data.edata['train_edge_observed_mask'],1)
    else:
        raise"no task "
    #先设置好了list共有节点数个list

    max_idx = max(src.max(), dst.max())
    print(src.min(),src.max(), dst.min(), dst.max())
    print("max_idx",max_idx)

    print(train_src.shape)
    print("Generate a set of neighbors for a train node>>>>>>>>")
    train_adj_list = init_adjacency_list(max_idx, train_src, train_dst, train_edge ,train_t)
    train_ngh_finder = CAWNeighborFinder(train_adj_list, bias=config["bias"], use_cache=config["ngh_cache"],sample_method=config["pos_sample"])

    print("Generate a set of neighbors for a full node>>>>>>>>")
    full_adj_list = init_adjacency_list(max_idx, src, dst, edge_idx ,t)
    full_ngh_finder = CAWNeighborFinder(full_adj_list, bias=config["bias"], use_cache=config["ngh_cache"],sample_method=config["pos_sample"])

    empty = np.zeros(Data.edata["edge_feat"].shape[1])[np.newaxis, :]

    empty = torch.zeros(Data.edata["edge_feat"].shape[1]).view(1,-1)
    edge_features = torch.vstack([empty, Data.edata["edge_feat"]])
    
    model = CAWConv([train_ngh_finder,full_ngh_finder], Data.ndata["feat"],  edge_features,agg=config["agg"],
                num_layers = config["n_layer"], use_time = config["time"]  , attn_agg_method = config["agg_method"],attn_mode = config["attn_mode"],
                n_head = config["attn_n_head"], drop_out = config["drop_out"] , pos_dim = config["pos_dim"] ,pos_enc = config["pos_enc"] ,
                num_neighbors = config["n_degree"] ,  walk_n_head = config["walk_n_head"] , walk_mutual = config["walk_mutual"] if config["walk_pool"] == 'attn' else False,
                walk_linear_out = config["walk_linear_out"] , walk_pool = config["walk_pool"],cpu_cores = config["cpu_cores"], verbosity = config["verbosity"])


    return model

def get_model(config,Data,task):
    if config["model"]=="TGAT":
        model = TGAT_init(config,Data,task)
    elif config["model"]=="CAW":
        model = CAW_init(config,Data,task)
    elif config["model"] == "Jodie" or config["model"] == "TGN" or config["model"] == "DeRep":
        model = TGN_init(config,Data,task)
    else:
        raise "error"

    return model




