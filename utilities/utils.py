
import numpy as np
import sys
import torch

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour) #True if pixel==palette[i] else False
        class_map = np.all(equality, axis=-1) #True if pixel==palette[i] in all channel else False
        semantic_map.append(class_map) #add into map list
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32) #append map into new dimension(1,512,512)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)

    colour_codes = np.array(palette)
    x = colour_codes[x.astype(np.uint8)]
    return x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message) #print to screen
        self.log.write(message) #print to logfile

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def find_bb(volume):
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 3
    # axis
    for i in range(img_shape[0]):
        img_slice_begin = volume[i, :, :]
        if np.sum(img_slice_begin) > 0:
            bb[0] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0] - 1 - i, :, :]
        if np.sum(img_slice_end) > 0:
            bb[1] = np.min([img_shape[0] - 1 - i + bb_extend, img_shape[0] - 1])
            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:, i, :]
        if np.sum(img_slice_begin) > 0:
            bb[2] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:, img_shape[1] - 1 - i, :]
        if np.sum(img_slice_end) > 0:
            bb[3] = np.min([img_shape[1] - 1 - i + bb_extend, img_shape[1] - 1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:, :, i]
        if np.sum(img_slice_begin) > 0:
            bb[4] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:, :, img_shape[2] - 1 - i]
        if np.sum(img_slice_end) > 0:
            bb[5] = np.min([img_shape[2] - 1 - i + bb_extend, img_shape[2] - 1])
            break

    return bb

def monitor_gradients(model, threshold=1000):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            if grad_mean == 0 or grad_std == 0 or torch.isnan(torch.tensor(grad_mean)) or torch.isnan(torch.tensor(grad_std)) or grad_mean > threshold or grad_std > threshold:
                print(f'{name}: grad mean={grad_mean}, grad std={grad_std}')



# 遍历数据集并打印样本信息
def print_dataset_content(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i} - Image Shape: {sample['image'].shape}, Mask Shape: {sample['mask'].shape}")

# 检查数据集是否为空
def check_dataset_empty(dataset):
    if len(dataset) == 0:
        print("Dataset is empty.")
    else:
        print("Dataset is not empty.")


# def load_pretrained_weights(model, pretrained_path):
#     # Load the pretrained state dictionary
#     pretrained_state_dict = torch.load(pretrained_path)
#     model_state_dict = model.state_dict()
#
#     # Create new state dictionary to load
#     new_state_dict = {}
#
#     # Iterate over the pretrained model's keys
#     for k in pretrained_state_dict.keys():
#         # Check if the key exists in the current model
#         if k in model_state_dict:
#             # Check if the shape of the weights match
#             if pretrained_state_dict[k].shape == model_state_dict[k].shape:
#                 new_state_dict[k] = pretrained_state_dict[k]
#             else:
#                 print(f"Size mismatch for {k}: pretrained weight shape {pretrained_state_dict[k].shape}, "
#                       f"model weight shape {model_state_dict[k].shape}. Skipping.")
#         else:
#             print(f"Key {k} is in the pretrained model but not in the current model. Skipping.")
#
#     # Load the new state dictionary into the model
#     model_state_dict.update(new_state_dict)
#     model.load_state_dict(model_state_dict)
#
#     print("Pretrained weights loaded (where possible).")
def load_pretrained_weights(model, pretrained_path):
    # Load the pretrained state dictionary
    pretrained_state_dict = torch.load(pretrained_path)
    model_state_dict = model.state_dict()

    # Create new state dictionary to load
    new_state_dict = {}

    # Iterate over the pretrained model's keys
    for k in pretrained_state_dict.keys():
        # Check if the key exists in the current model
        if k in model_state_dict:
            # Check if the shape of the weights match
            if pretrained_state_dict[k].shape == model_state_dict[k].shape:
                new_state_dict[k] = pretrained_state_dict[k]
            else:
                print(f"Size mismatch for {k}: pretrained weight shape {pretrained_state_dict[k].shape}, "
                      f"model weight shape {model_state_dict[k].shape}. Skipping.")
        else:
            print(f"Key {k} is in the pretrained model but not in the current model. Skipping.")

    # Load the new state dictionary into the model
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    # Print the total number of parameters in the model and pretrained model
    print('Total model_dict:', len(model_state_dict))
    print('Total pretrained_dict:', len(pretrained_state_dict))

    # Find and print not loaded keys
    not_loaded_keys = [k for k in pretrained_state_dict.keys() if k not in new_state_dict.keys()]
    print('Not loaded keys:', len(not_loaded_keys))

    print("Pretrained weights loaded (where possible).")
