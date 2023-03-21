import numpy as np
import pandas as pd
import os

def best_metric(config):
    save_dir= os.path.join("./metric",config["model"],config["dataset"])
    if config["task"] == "link_prediction":
        test_metric = np.loadtxt(os.path.join(save_dir,"link-test_metric.txt"))
        test_observed_metric = np.loadtxt(os.path.join(save_dir,"link-test_observed_metric.txt"))
        best_test = np.max(test_metric,axis=0)
        best_observed_metric = np.max(test_observed_metric,axis=0)

        return best_test,best_observed_metric


def pandas_write(path,data,columns):
    df = pd.DataFrame(data, columns=['col1', 'col2', 'col3'])
    df.to_csv('data.txt', sep='\t', index=False)

def max_pandas(path):
    df = pd.read_table(path, delimiter='\t')
    max_values = df.max()
    return max_values

