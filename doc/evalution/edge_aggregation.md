# MergeLayer
>`class lib.evaluator.MergeLayer(dim1, dim2, dim3, dim4, non_linear=True)`

Bases : `torch.nn.modules.module.Module`

Get the probability of forming an edge between the source node and the target node through two linear layers

Parameters
- dim1 : Dimensions of the first input vector
- dim2 : Dimensions of the second input vector
- dim3 : Project the dimension of dim1+dim2 to the dimension of dim3
- dim4 : Project the dimension of dim3 to the dimension of dim4
- non_linear : Whether to use the linear layer
---
>`forward(x1, x2)`

Parameters
- x1 `torch.tensor`: Representation of the source node
- x2 `torch.tensor`: Representation of the target node

**Return**
    **z** - The probability that a source node and a target node form an edge
    **Return type**         `torch.tensor`

# link_prediction_metric
>`lib.evaluator.link_prediction_metric(pos_prob,neg_prob,size=None)`

Parameters
- pos_prob : Dimensions of the first input vector
- neg_prob : Dimensions of the second input vector
- size : Project the dimension of dim1+dim2 to the dimension of dim3

**Return**
    **acc** - Describe the classification accuracy of the classifier ACC=(TP+TN)/(TP+FP+FN+TN)
    **Return type**         float

    **ap** - Obtained from the area of the PR curve and the X-axis, it is used to measure the detection of a class. The larger the AP, the better the detection of a single class
    **Return type**         float

    **f1** - The probability that a source node and a target node form an edge
    **Return type**         float
    
    **auc** - (Area under the Curve of ROC) is the area under the ROC curve, which is the standard for judging the pros and cons of the binary classification prediction model
    **Return type**         float
