import os
import math

import torch
import numpy as np
from tqdm import tqdm

from lib.dataloading import RandEdgeSampler
from lib.evaluator import link_prediction_metric
from lib.utils import get_logger,set_checkpoints_metric,numpy_from_mask

def eval_one_epoch(model,Edge_predictor, sampler, src, dst, ts,edges):
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
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)

            src_embed , target_embed= model(src_l_cut, dst_l_cut ,ts_l_cut , e_l_cut ,True)
            _ , background_embed = model(src_l_cut, dst_l_fake, ts_l_cut ,e_l_cut , False)

            pos_prob  = Edge_predictor(src_embed, target_embed).squeeze(dim = -1).sigmoid()
            neg_prob =  Edge_predictor(src_embed, background_embed).squeeze(dim = -1).sigmoid()

            metric = link_prediction_metric(pos_prob,neg_prob,size)
            
            for val_metric , x in zip([val_acc,val_ap,val_f1,val_auc],metric):
                val_metric.append(x)

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

def CAW_link_prediction(config_object,model_object,dataset,Edge_predict):
    
    # select dataset ,model,config,Edge_predictor 
    config = config_object
    model = model_object
    data = dataset
    Edge_predictor = Edge_predict

    checkpoints_path ,save_dir= set_checkpoints_metric(config,[config["agg_method"],config["attn_mode"],config["time"]])

    logger = get_logger([config["dataset"],config["model"],config["task"]])

    device = config['device']

    BATCH_SIZE = config['bs']

    src ,dst = data.edges()[0].numpy(),data.edges()[1].numpy()

    train_src,train_dst, train_t , train_edge = numpy_from_mask(data,data.edata['train_edge_observed_mask'],1)

    valid_src, valid_dst, valid_t, valid_edge  = numpy_from_mask(data,data.edata['valid_edge_mask'],1)
    valid_observed_src, valid_observed_dst, valid_observed_t, valid_observed_edge = numpy_from_mask(data,data.edata['valid_edge_observed_mask'],1)

    test_src, test_dst, test_t, test_edge = numpy_from_mask(data,data.edata['test_edge_mask'],1)
    test_observed_src, test_observed_dst, test_observed_t, test_observed_edge = numpy_from_mask(data,data.edata['test_edge_observed_mask'],1)

    model = model.to(device)
    Edge_predictor = Edge_predictor.to(device)

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
    idx_list = np.arange(num_instance)

    for epoch in range(config["n_epoch"]):
        
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        logger.info('start train {} epoch'.format(epoch))
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

            src_embed , target_embed= model(src_cut, dst_cut ,time_cut , edge_cut ,True)
            _ , background_embed = model(src_cut , dst_fake, time_cut ,edge_cut , False)

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
                
            
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(model, Edge_predictor,val_rand_sampler, valid_src, \
                                                            valid_dst, valid_t,valid_edge)
        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(model, Edge_predictor,nn_val_rand_sampler, valid_observed_src, \
                                                            valid_observed_dst, valid_observed_t,valid_observed_edge )

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

    for epoch in range(config["n_epoch"]):
        logger.info('start test {} epoch'.format(epoch))
        checkpoint = torch.load(checkpoints_path(epoch))
        model.load_state_dict(checkpoint['model'])
        Edge_predictor.load_state_dict(checkpoint['Edge_predictor'])
        model.set_state("eval")
        Edge_predictor.eval()

        test_acc, test_ap, test_f1, test_auc = eval_one_epoch(model, Edge_predictor,val_rand_sampler, test_src, \
                                                            test_dst, test_t,test_edge)
        nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(model, Edge_predictor,nn_test_rand_sampler, test_observed_src, \
                                                            test_observed_dst, test_observed_t, test_observed_edge)
        
        logger.info('test epoch: {}:'.format(epoch))
        logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {} , f1{}'.format(test_acc, test_auc, test_ap, test_f1))
        logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {} , f1{}'.format(nn_test_acc, nn_test_auc, nn_test_ap, nn_test_f1,))

        test_metric.append([test_acc, test_ap, test_f1, test_auc])
        test_observed_metric.append(nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc)
        
    np.savetxt(os.path.join(save_dir,"link-train_metirc.txt"),np.array(train_metirc))

    np.savetxt(os.path.join(save_dir,"link-valid_metric.txt"),np.array(valid_metric))
    np.savetxt(os.path.join(save_dir,"link-valid_observed_metric.txt"),np.array(valid_observed_metric))

    np.savetxt(os.path.join(save_dir,"link-test_metric.txt"),np.array(test_metric))
    np.savetxt(os.path.join(save_dir,"link-test_observed_metric.txt"),np.array(test_observed_metric))


