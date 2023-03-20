from ..evaluator import MergeLayer 
from .link_prediction import CAW_link_prediction,TGN_link_prediction,TGAT_link_prediction,DyGNN_link_prediction

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
    data = dataset
    if task == "link_prediction":
        Edge_predictor = MergeLayer(model.feat_dim, model.feat_dim, model.feat_dim, 1)
        if model_name == "CAW":
            Edge_predictor = MergeLayer(model.feat_dim, model.feat_dim, model.feat_dim, 1,not config["walk_linear_out"])
            CAW_link_prediction(config_object=config,model_object=model,dataset=data,Edge_predict=Edge_predictor)
        elif model_name == "TGN" or model_name == "Jodie" or model_name == "DyRep":
            TGN_link_prediction(config_object=config,model_object=model,dataset=data,Edge_predict=Edge_predictor)
        elif model_name == "TGAT":
            TGAT_link_prediction(config_object=config,model_object=model,dataset=data,Edge_predict=Edge_predictor)
        elif model_name == "DyGNN":
            DyGNN_link_prediction(config_object=config,model_object=model,dataset=data,Edge_predict=Edge_predictor)
        
    
     