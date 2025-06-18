"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2
import pickle
import logging
# import clip
import h5py
logger = logging.getLogger(__name__)



class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenizer
        
        loc_cap = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'id_mapping.json')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Dataset id
        with open(osp.join(loc_cap, '%s_ids.txt' % data_split), 'rb') as f:
            image_ids = f.readlines()
            self.image_ids = [int(x.strip()) for x in image_ids]

        # self.all_data = h5py.File(loc_cap+'/' + data_name + '_all_align.hdf5','r')   #打开h5文件
        # self.all_data = h5py.File(loc_cap+'/' + data_name + '_all_align_R101_new1.hdf5','r')   #打开h5文件
        # self.all_data = h5py.File('/home/zy/data_temp/coco_all_align.hdf5','r')   #打开h5文件
        # self.all_data = h5py.File('/home/zy/data_temp/f30k_all_align_R101_butd2.hdf5','r')   #打开h5文件
        self.all_data = h5py.File('/home/zy/dev/datasets/f30k/precomp/f30k_all_align.hdf5', 'r')

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.image_ids)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):

            
        # handle the image redundancy
        img_index = index // self.im_div
        image_id = self.image_ids[img_index]
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time)
        target = process_caption(self.tokenizer, caption_tokens, self.train)
        
        rg_features = self.all_data['%d_features' % image_id][:]
        rg_boxes = self.all_data['%d_boxes' % image_id][:]
        # print(rg_boxes.size)
        raw_sizes = self.all_data['%d_size' % image_id][:]
        raw_sizes = raw_sizes[:, ::-1]
        raw_sizes = np.concatenate([raw_sizes, raw_sizes], axis=1)
        relative_boxes = rg_boxes / raw_sizes

        gd_features = torch.tensor(self.all_data['%d_grids' % image_id][:])
        al_graph = self.all_data['%d_mask' % image_id][:]

        return rg_features.astype(np.float32), relative_boxes.astype(np.float32), gd_features, al_graph.astype(np.float32), target, index, img_index

    def __len__(self):
        return self.length



def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    rg_features, relative_boxes, gd_features, al_graphs, captions, ids, im_id = zip(*data)
    max_detections = 50

    #     # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    rg_lengths = [len(rgs) for rgs in rg_features]
    rg_features_batch = torch.zeros(len(rg_features), max_detections, rg_features[0].shape[-1])
    relative_boxes_batch = torch.zeros(len(relative_boxes), max_detections, 4)
    al_graphs_batch = torch.zeros(len(al_graphs), max_detections, 49)

    for i, rgs in enumerate(rg_features):
        end = rg_lengths[i]
        assert end == relative_boxes[i].shape[0] == al_graphs[i].shape[0]
        delta = max_detections - end
        if delta >= 0:
            rg_features_batch[i, :end] = torch.tensor(rgs[:end])
            relative_boxes_batch[i, :end] = torch.tensor(relative_boxes[i][:end])
            al_graphs_batch[i, :end] = torch.tensor(al_graphs[i][:end])
        elif delta < 0:
            rg_features_batch[i,:] = torch.tensor(rgs[:max_detections])
            relative_boxes_batch[i, :] = torch.tensor(relative_boxes[i][:max_detections])
            al_graphs_batch[i, :] = torch.tensor(al_graphs[i][:max_detections])

    gd_features_batch = torch.stack(gd_features, 0)
    rg_lengths = torch.Tensor(rg_lengths)
    rg_lengths = torch.clamp(rg_lengths, max=max_detections)
    lengths = torch.Tensor(lengths)

    return rg_features_batch, rg_lengths, relative_boxes_batch, gd_features_batch, al_graphs_batch, targets, lengths, ids, im_id


def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True, clip_model = None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False

    dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)

    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):

    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
