from torch.utils.data import Dataset
import glob
import numpy as np
import torch
import pickle
from tqdm import tqdm

class TMED2_4_Class_Semi_Weighted(Dataset):
    def __init__(self, data_root, transforms, subset, norm, rtn_id, apply_mask, sup_fraction, mu):

        self.avc_classes = ['PLAX', 'PSAX', 'A2C', 'A4C']
        self.class_idx_dict = self.class_to_idx()

        self.transforms = transforms
        self.data_root = data_root + subset
        self.norm = norm
        self.rtn_id = rtn_id
        self.mu = mu
        self.sup_fraction = sup_fraction

        with open(data_root + 'exclude_bad_mask.pkl', 'rb') as f:
            exclude_list = pickle.load(f)

        # select a fraction of frames where backmix is applied.
        self.frame_path_list = sorted(self.exclude_examples(glob.glob(data_root + subset + '/*'), exclude_list, subset))
        self.mask_path_list = sorted(self.exclude_examples(glob.glob(data_root + 'masking2/' + subset + '/*'), exclude_list, subset))
        every_N_frames = int(1/sup_fraction)
        print('supervising {} of frames. that is every {} frame'.format(sup_fraction, every_N_frames))
        self.frame_path_list_skips = [self.frame_path_list[i] if i % every_N_frames == 0 else 'SKIP' for i in range(len(self.frame_path_list))] # 20% of data have masks
        self.mask_path_list_skips = [self.mask_path_list[i] if i % every_N_frames == 0 else 'SKIP' for i in range(len(self.mask_path_list))] # 20% of data have masks

        self.frame_to_sample = [f for f in self.frame_path_list_skips if f != 'SKIP']
        self.masks_to_sample = [f for f in self.mask_path_list_skips if f != 'SKIP']

        assert len(self.frame_path_list) == len(self.mask_path_list)

        print('{} set has {} images'.format(subset, len(self.frame_path_list)))

        if subset == 'train':
            self.apply_mask = apply_mask
        else:
            self.apply_mask = None
        print('{} masking: {}'.format(subset, self.apply_mask))

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

        # load mask if available
        if self.mask_path_list_skips[index] == 'SKIP':
            skip = True
        else:
            mask = np.load(self.mask_path_list_skips[index]).astype(np.uint8)
            skip = False

        # augmentation
        if self.transforms is not None:
            if self.apply_mask is not None and not skip:
                augmented = self.transforms(image=frame, mask=mask)
                frame = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transforms(image=frame)
                frame = augmented['image']

        frame = frame.astype(np.float32)

        # apply mask to frame (if applicable)
        if self.apply_mask is None:
            pass
        elif self.apply_mask == 'random_bg':
            if skip:
                pass
            else:
                # select a random frame which has a mask
                random_idx = np.random.randint(0, len(self.frame_to_sample))
                random_frame = np.load(self.frame_to_sample[random_idx]).astype(np.uint8)
                random_mask = np.load(self.masks_to_sample[random_idx]).astype(np.uint8)
                # black out the part that is supposed to be the triangle.
                random_frame[random_mask == 1] = np.zeros_like(random_frame[random_mask == 1])
                # add random background to frame.
                frame[mask == 0] = random_frame[mask == 0]

        # format
        frame = np.expand_dims(frame, axis=0)
        frame = np.clip(frame, 0, 255) / 255
        if self.norm == 'z':
            if np.sqrt(frame.var()) == 0 or np.isnan(np.sqrt(frame.var())):
                print('caught mask with std 0 or nan ', id_)
            frame = (frame - frame.mean()) / np.sqrt(frame.var())
        frame = torch.from_numpy(frame).float()
        label = torch.tensor(label).long()

        # wBackMix weights
        if skip:
            loss_weight = torch.tensor(1 - (self.mu * self.sup_fraction))
        else:
            loss_weight = torch.tensor(1 + (self.mu * (1 - self.sup_fraction)))

        return frame, label, loss_weight, id_

