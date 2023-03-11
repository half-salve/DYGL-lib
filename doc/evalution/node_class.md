# MLP
>`class lib.evaluator.MLP(dim, drop=0.3)`

Bases : `torch.nn.modules.module.Module`

Let the node representation go through two linear layers to get the classification result of the node

**Parameters**

- dim1 : Dimensions of the first input vector
- dim2 : Dimensions of the second input vector
- dim3 : Project the dimension of dim1+dim2 to the dimension of dim3
- dim4 : Project the dimension of dim3 to the dimension of dim4
- non_linear : Whether to use the linear layer

>`forward(x)`

**Parameters**

- x : node representation

**Return**

    **Return type**         `torch.tensor`