import lib

import argparse
import sys
import os
import math

import torch
import numpy as np
from tqdm import tqdm

from lib.dataloading import RandEdgeSampler
from lib.evaluator import MergeLayer 
from lib.utils import get_logger,data_from_mask

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import dgl
def get_data(name):
    if name == "wikipedia":
        Data = lib.data.WikipediaDataset()[0]
    if name == "Reddit":
        Data = lib.data.RedditDataset()[0]
    if name == "Mooc":
        Data = lib.data.MOOCDataset()[0]
    if name == "Mooc":
        pass
    return Data

def eval_one_epoch(model,Edge_predictor, sampler, src, dst, ts):
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
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)

            src_embed = model(src_l_cut, ts_l_cut)
            target_embed = model(dst_l_cut, ts_l_cut)
            background_embed = model(dst_l_fake,ts_l_cut)

            pos_prob  = Edge_predictor(src_embed, target_embed).squeeze(dim = -1).sigmoid()
            neg_prob =  Edge_predictor(src_embed, background_embed).squeeze(dim = -1).sigmoid()

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

parser = argparse.ArgumentParser(
    'The code used to  node classification')

# select dataset and training model
parser.add_argument('--dataset', type=str, default='wikipedia', help='data sources to use, try wikipedia or reddit',
                    choices=["wikipedia" ,"Reddit","Mooc","redditlink"], )
parser.add_argument( '--model', type=str, default='TGAT', help='select model that you want to use',
                    choices=["TGAT,CAW"], )
parser.add_argument( '--task', type=str, default='link_prediction', help='select task that you want to use',
                    choices=["link_prediction,node_classification"], )
# general training hyper-parameters
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')

# parameters controlling computation settings but not affecting results in general
parser.add_argument('--seed', type=int, default=2023, help='random seed for all randomized algorithms')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--exp_id', type=int, default=0, help='')
args = parser.parse_args()

# select dataset and training mode
config=lib.ConfigParser(args)

saved_checkpoints_dir = f'./saved_checkpoints/{config["model"]}/{config["dataset"]}-{config["agg_method"]}-{config["attn_mode"]}/'
if not os.path.exists(saved_checkpoints_dir):
    os.makedirs(saved_checkpoints_dir)
checkpoints_path = lambda epoch: f'{saved_checkpoints_dir}/{config["prefix"]}-{epoch}.pth'

logger = get_logger([config["dataset"],config["model"],config["task"]])

device = config['device']

BATCH_SIZE = config['bs']

Data = get_data(config["dataset"])

src ,dst = Data.edges()[0] ,Data.edges()[1]
#现在的情况是 ['train_edge_observed_mask']是[,1],[Data.edata['valid_edge_mask']]是-1，但是结果刚好相反，train中的是[-1]
t = Data.edata["time"]
print(Data.edata['train_edge_observed_mask'].shape,Data.edata['valid_edge_mask'].shape)

train_src,train_dst, train_t , train_edge = data_from_mask(Data,Data.edata['train_edge_observed_mask'])

valid_src, valid_dst, valid_t, valid_edge  = data_from_mask(Data,Data.edata['valid_edge_mask'])
valid_observed_src, valid_observed_dst, valid_observed_t, valid_observed_edge = data_from_mask(Data,Data.edata['valid_edge_observed_mask'])

test_src, test_dst, test_t, test_edge = data_from_mask(Data,Data.edata['test_edge_mask'])
test_observed_src, test_observed_dst, test_observed_t, test_observed_edge = data_from_mask(Data,Data.edata['test_edge_observed_mask'])

print(train_src)
print(train_src.shape,valid_src.shape,src.shape)

model = lib.get_model(config,Data,config["task"]).to(device)

Edge_predictor = MergeLayer(model.feat_dim, model.feat_dim, model.feat_dim, 1).to(device)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam([{'params':model.parameters()},
                             {'params':Edge_predictor.parameters()},],
                             lr=config["learning_rate"])

train_rand_sampler = RandEdgeSampler(train_src, train_dst)

val_rand_sampler = test_rand_sampler = RandEdgeSampler(src, dst)

nn_val_rand_sampler = RandEdgeSampler(valid_observed_src, valid_observed_src)
nn_test_rand_sampler = RandEdgeSampler(test_observed_src, test_observed_src)

num_instance = train_src.shape[0]
num_batch = math.ceil(num_instance / BATCH_SIZE)

print('num of training instances: {}'.format(num_instance))
print('num of batches per epoch: {}'.format(num_batch))

num_params = sum(param.numel() for param in model.parameters())
logger.info("num_params:={} params={} M".format(num_params,num_params/1e6))

train_metirc =[]
valid_metric = []
valid_observed_metric = []

for epoch in range(config["n_epoch"]):
    
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    idx_list = torch.randperm(num_instance)
    logger.info('start train {} epoch'.format(epoch))
    for k in tqdm(range(num_batch)):
        src_idx = k * BATCH_SIZE

        dst_idx = min(num_instance - 1, src_idx + BATCH_SIZE)

        src_cut , dst_cut = train_src[src_idx:dst_idx], train_dst[src_idx:dst_idx]

        time_cut = train_t[src_idx:dst_idx]

        size = src_cut.shape[0]

        _ , dst_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        model.set_state("train")
        Edge_predictor.train()

        src_embed = model(src_cut, time_cut)

        target_embed = model(dst_cut, time_cut)
        background_embed = model(dst_fake,time_cut)

        pos_prob  = Edge_predictor(src_embed, target_embed).squeeze(dim = -1).sigmoid()

        neg_prob =  Edge_predictor(src_embed, background_embed).squeeze(dim = -1).sigmoid()

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.set_state("eval")
            Edge_predictor.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])

            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))
        
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch(model, Edge_predictor,val_rand_sampler, valid_src, \
                                                        valid_dst, valid_t)
    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(model, Edge_predictor,nn_val_rand_sampler, valid_observed_src, \
                                                        valid_observed_dst, valid_observed_t, )

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

test_metric = []
test_observed_metric =[]

save_dir= os.path.join("./metric",config["model"],config["dataset"])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for epoch in range(config["n_epoch"]):
    logger.info('start test {} epoch'.format(epoch))
    checkpoint = torch.load(checkpoints_path(epoch))
    model.load_state_dict(checkpoint['model'])
    Edge_predictor.load_state_dict(checkpoint['Edge_predictor'])
    model.set_state("eval")
    Edge_predictor.eval()

    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(model, Edge_predictor,val_rand_sampler, test_src, \
                                                        test_dst, test_t)
    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(model, Edge_predictor,nn_test_rand_sampler, test_observed_src, \
                                                        test_observed_dst, test_observed_t, )
    
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


