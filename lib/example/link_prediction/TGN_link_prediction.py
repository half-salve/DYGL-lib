import lib

import argparse
import sys
import os
import math

import torch
import numpy as np
from tqdm import tqdm

from lib.dataloading import RandEdgeSampler
from lib.evaluator import MergeLayer ,link_prediction_metric
from lib.utils import get_logger,set_checkpoints_metric,numpy_from_mask
from lib.utils import get_data

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


torch.manual_seed(0)
np.random.seed(0)

def eval_edge_prediction(model,Edge_predictor, sampler, src, dst, ts,edges):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        model.set_state("eval")
        Edge_predictor.eval()

        TEST_BATCH_SIZE=128
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        print("start evaluation one epoch<<<<<")
        for k in tqdm(range(num_test_batch)):
            
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = edges[s_idx:e_idx]

            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)

            src_embed, target_embed,background_embed= model(source_nodes=src_l_cut,destination_nodes = dst_l_cut,
                                                 negative_nodes=dst_l_fake, edge_times = ts_l_cut, edge_idxs = e_l_cut )

            pos_prob  = Edge_predictor(src_embed, target_embed).squeeze(dim = -1).sigmoid()
            neg_prob =  Edge_predictor(src_embed, background_embed).squeeze(dim = -1).sigmoid()

            metric = link_prediction_metric(pos_prob,neg_prob,size)
            
            for val_metric , x in zip([val_acc,val_ap,val_f1,val_auc],metric):
                val_metric.append(x) 

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)


### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')


parser.add_argument('--dataset', type=str, default='wikipedia', help='data sources to use, try wikipedia or reddit',
                    choices=["wikipedia" ,"Reddit","Mooc","redditlink"], )
parser.add_argument( '--model', type=str, default='TGN', help='select model that you want to use',
                    choices=["Jodie,TGN,DyRep"], )
parser.add_argument( '--task', type=str, default='link_prediction', help='select task that you want to use',
                    choices=["link_prediction,node_classification"], )
# general training hyper-parameters
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')

# parameters controlling computation settings but not affecting results in general
parser.add_argument('--seed', type=int, default=2023, help='random seed for all randomized algorithms')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--exp_id', type=int, default=0, help='')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

config=lib.ConfigParser(args)
checkpoints_path ,save_dir= set_checkpoints_metric(config,[config["prefix"],config["embedding_module"],config["memory_updater"]])

### set up logger
logger = get_logger([config["dataset"],config["model"],config["task"]])

## get batch_size and dataset
BATCH_SIZE = config['bs']
Data = get_data(config["dataset"])
src ,dst = Data.edges()
t = Data.edata["time"]

### Extract data for training, validation and testing

train_src,train_dst, train_t , train_edge = numpy_from_mask(Data,Data.edata['train_edge_observed_mask'])

valid_src, valid_dst, valid_t, valid_edge  = numpy_from_mask(Data,Data.edata['valid_edge_mask'])
valid_observed_src, valid_observed_dst, valid_observed_t, valid_observed_edge = numpy_from_mask(Data,Data.edata['valid_edge_observed_mask'])

test_src, test_dst, test_t, test_edge = numpy_from_mask(Data,Data.edata['test_edge_mask'])
test_observed_src, test_observed_dst, test_observed_t, test_observed_edge = numpy_from_mask(Data,Data.edata['test_edge_observed_mask'])

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_src, train_dst)

val_rand_sampler = test_rand_sampler = RandEdgeSampler(src, dst)

nn_val_rand_sampler = RandEdgeSampler(valid_observed_src, valid_observed_src)
nn_test_rand_sampler = RandEdgeSampler(test_observed_src, test_observed_src)

# Set device
device = config['device']

# Initialize Model
model = lib.get_model(config,Data,config["task"]).to(device)
Edge_predictor = MergeLayer(model.feat_dim, model.feat_dim, model.feat_dim, 1).to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam([{'params':model.parameters()},
                             {'params':Edge_predictor.parameters()},],
                             lr=config["learning_rate"])

num_instance = train_src.shape[0]
num_batch = math.ceil(num_instance / BATCH_SIZE)

print('num of training instances: {}'.format(num_instance))
print('num of batches per epoch: {}'.format(num_batch))

num_params = sum(param.numel() for param in model.parameters())
logger.info("num_params:={} params={} M".format(num_params,num_params/1e6))

idx_list = np.arange(num_instance)

train_metirc =[]
valid_metric = []
valid_observed_metric = []

for epoch in range(config["n_epoch"]):
    
    acc, ap, f1, auc, m_loss = [], [], [], [], []

    logger.info('start train {} epoch'.format(epoch))
        # Reinitialize memory of the model at the start of each epoch
    if  config["use_memory"]:
        model.memory.__init_memory__()
    for k in tqdm(range(num_batch)):
        src_idx = k * BATCH_SIZE
        dst_idx = min(num_instance - 1, src_idx + BATCH_SIZE)

        batch_idx = idx_list[src_idx:dst_idx ]
        src_cut , dst_cut = train_src[batch_idx], train_dst[batch_idx]
        time_cut , edge_cut = train_t[batch_idx], train_edge[batch_idx]

        size = src_cut.shape[0]

        _ , dst_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        model.set_state("train")
        Edge_predictor.train()

        src_embed, target_embed,background_embed= model(source_nodes=src_cut,destination_nodes = dst_cut,
                                                 negative_nodes=dst_fake, edge_times = time_cut, edge_idxs = edge_cut)

        pos_prob  = Edge_predictor(src_embed, target_embed).squeeze(dim = -1).sigmoid()

        neg_prob =  Edge_predictor(src_embed, background_embed).squeeze(dim = -1).sigmoid()

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.set_state("eval")
            Edge_predictor.eval()

            metric = link_prediction_metric(pos_prob,neg_prob,size)
            for train_metric , x in zip([acc,ap,f1,auc],metric):
                train_metric.append(x) 
            m_loss.append(loss.item())

        if config["use_memory"]:
            model.memory.detach_memory()

    if  config["use_memory"]:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
        train_memory_backup = model.memory.backup_memory()    
    val_acc, val_ap, val_f1, val_auc = eval_edge_prediction(model, Edge_predictor,val_rand_sampler, valid_src, \
                                                        valid_dst, valid_t,valid_edge)
    
    if  config["use_memory"]:
        val_memory_backup = model.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        model.memory.restore_memory(train_memory_backup)

    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_edge_prediction(model, Edge_predictor,nn_val_rand_sampler, valid_observed_src, \
                                                        valid_observed_dst, valid_observed_t,valid_observed_edge )

    if  config["use_memory"]:
      # Restore memory we had at the end of validation
        model.memory.restore_memory(val_memory_backup)


    train_metirc.append([np.mean(acc), np.mean(ap) , np.mean(f1), np.mean(auc), np.mean(m_loss)])
    valid_observed_metric.append([nn_val_acc, nn_val_ap ,nn_val_f1 ,nn_val_auc ])
    valid_metric.append([val_acc, val_ap, val_f1, val_auc])


    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(train_metirc[epoch][0], val_acc, nn_val_acc))
    logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(train_metirc[epoch][3], val_auc, nn_val_auc))
    logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(train_metirc[epoch][2], val_ap, nn_val_ap))
    logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(train_metirc[epoch][1], val_f1, nn_val_f1))


    state = {"model":model.state_dict(),
             "Edge_predictor":Edge_predictor.state_dict()
    }
    torch.save(state, checkpoints_path(epoch))


# Training has finished, we have loaded the best model, and we want to backup its current
# memory (which has seen validation edges) so that it can also be used when testing on unseen
# nodes
test_metric = []
test_observed_metric =[]

save_dir= os.path.join("./metric",config["model"],config["dataset"])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for epoch in range(config["n_epoch"]):

    if  config["use_memory"]:
        model.memory.__init_memory__()

    logger.info('start test {} epoch'.format(epoch))
    checkpoint = torch.load(checkpoints_path(epoch))
    model.load_state_dict(checkpoint['model'])
    Edge_predictor.load_state_dict(checkpoint['Edge_predictor'])
    model.set_state("eval")
    Edge_predictor.eval()

    if config["use_memory"]:
        val_memory_backup = model.memory.backup_memory()

    test_acc, test_ap, test_f1, test_auc = eval_edge_prediction(model, Edge_predictor,val_rand_sampler, test_src, \
                                                        test_dst, test_t,test_edge)
    
    if config["use_memory"]:
        model.memory.restore_memory(val_memory_backup)
    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_edge_prediction(model, Edge_predictor,nn_test_rand_sampler, test_observed_src, \
                                                        test_observed_dst, test_observed_t, test_observed_edge)
    
    logger.info('test epoch: {}:'.format(epoch))
    logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {} , f1{}'.format(test_acc, test_auc, test_ap, test_f1))
    logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {} , f1{}'.format(nn_test_acc, nn_test_auc, nn_test_ap, nn_test_f1))

    test_metric.append([test_acc, test_ap, test_f1, test_auc])

    test_observed_metric.append([nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc])

    
np.savetxt(os.path.join(save_dir,"train_metirc.txt"),np.array(train_metirc))

np.savetxt(os.path.join(save_dir,"valid_metric.txt"),np.array(valid_metric))
np.savetxt(os.path.join(save_dir,"valid_observed_metric.txt"),np.array(valid_observed_metric))

np.savetxt(os.path.join(save_dir,"test_metric.txt"),np.array(test_metric))
np.savetxt(os.path.join(save_dir,"test_observed_metric.txt"),np.array(test_observed_metric))