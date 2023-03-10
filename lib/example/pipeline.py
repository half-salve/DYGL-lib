from lib.evaluator import MergeLayer 


def train_and_test(task = None,model_name = None,dataset=None,model_object=None ,config_object =None,):
    """
        Args:
            task(str): task name
            model_name(str): model name
            dataset_name(str): dataset name
            config_object(object): config 
            model_object()object): model
    """
    
    config = config_object
    model =model_object
    if task == "link_prediction":
        Edge_predictor = MergeLayer(model.feat_dim, model.feat_dim, model.feat_dim, 1)
     