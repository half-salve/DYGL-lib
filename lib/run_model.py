"""
训练并评估单一模型的脚本
"""

import argparse
from mtlib.utils import  add_general_args
from mtlib.config import DGCONFIG,ConfigParser
from mtlib.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
from mtlib.data import normalDataset
from mtlib.utils import temporal_signal_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='link_prediction', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='CAW', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file address of config file')
    parser.add_argument('--saved_file', type=str,
                        default=None, help='the file address of save file')
    parser.add_argument('--train', type=bool, default=True,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()

    config = DGCONFIG(args=args,config_file=args.config_file,other_args=None)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = 0
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config,save_dir=config.get("saved_file",None))
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = normalDataset("wikipedia","D:\GNN_train\Data")
    # 转换数据，并划分数据集
    train_ratio = config.get('train_ratio',0.7)
    valid_ratio = config.get('train_ratio',0.85)
    train_data, valid_data, test_data = temporal_signal_split(dataset,train_ratio,valid_ratio)
    node_features = dataset.node_features
    edge_features = dataset.edge_features

    # 加载执行器
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(args=config, node_feature=node_features,edge_features=edge_features)

    executor = get_executor(args=config, model=model, data_feature=data_feature)
    # 训练
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)