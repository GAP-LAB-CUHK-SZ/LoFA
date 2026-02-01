from einops import rearrange

def lora_formatting(layer_nums, layer_types):
    block_name_list = []
    for depth in range(layer_nums):
        for module_name in layer_types:
            block_name_list.append(f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}")
    return block_name_list

def lora_ckpt_formatting(lora_A, lora_B, block_name_list, rank=1):
    lora_A = rearrange(lora_A, "B L (r c) -> B L c r", r=rank)
    lora_B = rearrange(lora_B, "B L (r c) -> B L r c", r=rank)
    state_dict = {}
    for i, module_name in enumerate(block_name_list):
        state_dict[module_name + ".lora_A.default.weight"] = lora_A[:, i, :, :]
        state_dict[module_name + ".lora_B.default.weight"] = lora_B[:, i, :, :]
    
    return state_dict