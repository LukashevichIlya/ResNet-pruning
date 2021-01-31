import copy


def make_idx_dict(model, counter, array, dictionary):
    for m_idx, m_k in enumerate(model._modules.keys()):
        new_array = copy.deepcopy(array)
        if len(model._modules[m_k]._modules.keys()):
            new_array.append(m_k)
            counter, dictionary = make_idx_dict(model._modules[m_k], counter, new_array, dictionary)
        else:
            new_array.append(m_k)
            counter = counter + 1
            dictionary[counter] = new_array
    return counter, dictionary


def get_layer_from_idx(model, idx_dict, idx):
    if len(idx_dict[idx]) == 1:
        return model._modules[idx_dict[idx][0]]
    m_idx = idx_dict[idx].pop(0)
    return get_layer_from_idx(model._modules[m_idx], idx_dict, idx)


def set_layer_to_idx(model, idx_dict, idx, layer):
    if len(idx_dict[idx]) == 1:
        model._modules[idx_dict[idx][0]] = layer
    else:
        m_idx = idx_dict[idx].pop(0)
        set_layer_to_idx(model._modules[m_idx], idx_dict, idx, layer)
