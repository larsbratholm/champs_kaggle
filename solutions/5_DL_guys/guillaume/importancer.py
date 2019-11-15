import torch
import numpy as np

tags = {}
selection = []

def reset_tags():
    global tags
    tags = {}

def get_tags():
    global tags
    return tags

def select_tags(select_tags):
    global selection
    global tags
    for tag in select_tags:
        if tag not in tags:
            raise ValueError(f"Tag {tag} not in tags")
    selection = select_tags

def tag(tensor, tag_name, tensor_slice = [], inplace = True):
    global tags
    global selection
    global MODE
    
    if not inplace and tensor_slice:
        raise ValueError("If inplace is False, the slice must be empty (the full tensor will be returned")
    
    full_tensor_slice = tuple([slice(None)] + list(tensor_slice))
    
    if tag_name not in tags:
        tags[tag_name] = {'size' : list(tensor.size()), 'slice' : tensor_slice}

    if tag_name in selection:
        permute_shape = tensor.size(0)
        if inplace:
            tensor[full_tensor_slice] = tensor[full_tensor_slice][torch.tensor(np.random.permutation(permute_shape))]
        else:
            return tensor[full_tensor_slice][torch.tensor(np.random.permutation(permute_shape))]
        
    else:
        if inplace:
            pass
        else:
            return tensor
    