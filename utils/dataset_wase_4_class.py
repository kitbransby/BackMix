from torch.utils.data import Dataset
import glob
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

class WASE_4_Class(Dataset):
    def __init__(self, data_root, transforms, subset, norm, rtn_id):

        self.avc_classes = ['PLAX', 'PSAX', 'A2C', 'A4C']
        self.class_idx_dict = self.class_to_idx()

        self.transforms = transforms
        self.data_root = data_root + subset
        self.norm = norm
        self.rtn_id = rtn_id

        with open(data_root + 'exclude_bad_mask.pkl', 'rb') as f:
            exclude_list = pickle.load(f)

        self.frame_path_list = sorted(self.exclude_examples(glob.glob(data_root + subset + '/*'), exclude_list, subset))
        self.mask_path_list = sorted(self.exclude_examples(glob.glob(data_root + 'masking2/' + subset + '/*'), exclude_list, subset))
        assert len(self.frame_path_list) == len(self.mask_path_list)

        print('{} set has {} images'.format(subset, len(self.frame_path_list)))

    def __len__(self):
        return len(self.frame_path_list)

    def load_file_list(self, file_list):
        return [np.load(f) for f in tqdm(file_list)]

    def exclude_examples(self, file_list, exclude_list, subset):
        updated_file_list = [f for f in file_list if f.split('/')[-1] not in exclude_list]
        print('excluding {} {} examples.'.format( len(file_list) - len(updated_file_list), subset))
        return updated_file_list

    def class_to_idx(self):
        class_idx_dict = {}
        for i, cls in enumerate(self.avc_classes):
            class_idx_dict[cls] = i
        return class_idx_dict

    def __getitem__(self, index):

        # load frame, label, id, noise
        file_name = self.frame_path_list[index]
        frame = np.load(file_name).astype(np.uint8)
        label = self.class_idx_dict[file_name.split('_')[-1].split('.')[0]]
        id_ = file_name.split('/')[-1]

        # augmentation
        if self.transforms is not None:
            augmented = self.transforms(image=frame)
            frame = augmented['image']

        frame = frame.astype(np.float32)

        # format
        frame = np.expand_dims(frame, axis=0)
        frame = np.clip(frame, 0, 255) / 255
        if self.norm == 'z':
            frame = (frame - frame.mean()) / np.sqrt(frame.var())
        frame = torch.from_numpy(frame).float()
        label = torch.tensor(label).long()

        if self.rtn_id:
            return frame, label, id_
        else:
            return frame, label
