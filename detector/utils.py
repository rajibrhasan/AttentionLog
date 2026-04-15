import torch 
import numpy as np

def process_attn(attention, rng, attn_func):
    heatmap = np.zeros((len(attention), attention[0].shape[1]))
    for i, attn_layer in enumerate(attention):
        attn_layer = attn_layer.to(torch.float32).numpy()

        if "sum" in attn_func:
            last_token_attn_to_inst = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            attn = last_token_attn_to_inst
        
        elif "max" in attn_func:
            last_token_attn_to_inst = np.max(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            attn = last_token_attn_to_inst

        else: raise NotImplementedError
            
        last_token_attn_to_inst_sum = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
        last_token_attn_to_data_sum = np.sum(attn_layer[0, :, -1, rng[1][0]:rng[1][1]], axis=1)

        if "normalize" in attn_func:
            epsilon = 1e-8
            heatmap[i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
        else:
            heatmap[i, :] = attn

    heatmap = np.nan_to_num(heatmap, nan=0.0)

    return heatmap


def process_attn_prefill(attention_maps, rng):
    """Process prefill attention: last data token → data tokens, max aggregation.

    Args:
        attention_maps: list of per-layer tensors, each [1, heads, seq_len, seq_len]
        rng: ((inst_start, inst_end), (data_start, data_end))

    Returns:
        heatmap: [num_layers, num_heads] with max attention to any data token
    """
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1]
    heatmap = np.zeros((num_layers, num_heads))

    data_start, data_end = rng[1]
    query_pos = data_end - 1  # last data token

    for i, attn_layer in enumerate(attention_maps):
        attn_layer = attn_layer.to(torch.float32).numpy()
        # attention from last data token to all data tokens: [heads, data_len]
        attn_to_data = attn_layer[0, :, query_pos, data_start:data_end]
        data_len = attn_to_data.shape[1]

        # max attention to any data token
        heatmap[i, :] = np.max(attn_to_data, axis=1)

    heatmap = np.nan_to_num(heatmap, nan=0.0)
    return heatmap


def calc_attn_score(heatmap, heads):
    score = np.mean([heatmap[l, h] for l, h in heads], axis=0)
    return score

