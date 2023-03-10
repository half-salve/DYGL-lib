from ..evaluator import MergeLayer 
from .link_prediction import CAW_link_prediction,TGN_link_prediction,TGAT_edge_pipeline

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
        if model_name == "TGN" or model_name == "Jodie" or model_name == "DeRep":
            TGN_link_prediction(config_object=config,model_object=model,dataset=data,Edge_predict=Edge_predictor)
        if model_name == "TGAT":
            TGAT_edge_pipeline(config_object=config,model_object=model,dataset=data,Edge_predict=Edge_predictor)
    
     