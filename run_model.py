import lib

import argparse
import sys

from lib.utils import get_data
from lib.example import train_and_test
parser = argparse.ArgumentParser('training')

parser.add_argument('--dataset', type=str, default='UCI', help='data sources to use, try wikipedia or reddit',
                    choices=["wikipedia" ,"Reddit","Mooc","redditlink","LastFM","UCI",], )
parser.add_argument( '--model', type=str, default='DyGNN', help='select model that you want to use',
                    choices=["Jodie","TGN","DyRep","TGAT","CAW","DyGNN"], )
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

### set up logger
# logger = get_logger([config["dataset"],config["model"],config["task"]])
## get and dataset
Data = get_data(config["dataset"],config["model"])

# Initialize Model
model = lib.get_model(config,Data,config["task"])

train_and_test (task = config["task"],model_name = config["model"],dataset=Data,model_object=model,config_object =config)
