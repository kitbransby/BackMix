from torch.utils.data import Dataset
import glob
import numpy as np
import torch
import pickle
from tqdm import tqdm

class TMED2_4_Class(Dataset):
    def __init__(self, data_root, transforms, subset, norm, rtn_id, apply_mask):

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

        # apply augmentation only at train time
        if subset == 'train':
            self.apply_mask = apply_mask
            self.noise_array = self.get_noise(self.apply_mask)
        else:
            self.apply_mask = None
        print('{} masking: {}'.format(subset, self.apply_mask))

    def __len__(self):
        return len(self.frame_path_list)

    def get_noise(self, noise_type):
        if noise_type == 'block_noise':
            noise = np.random.uniform(0, 30, 112 * 112)
        elif noise_type in ['shuffle_bg', 'block_black', 'bokeh', 'random_bg']:
            noise = None
        elif noise_type is None:
            noise = None
        return noise

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
        if self.apply_mask is not None:
            mask = np.load(self.mask_path_list[index]).astype(np.uint8)
            noise = self.noise_array

        # augmentation
        if self.transforms is not None:
            if self.apply_mask is not None:
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
        elif self.apply_mask == 'block_noise':
            np.random.shuffle(noise)
            noise = noise.reshape((112, 112))
            frame[mask == 0] = noise[mask == 0]
        elif self.apply_mask == 'shuffle_bg':
            bg = frame[mask == 0]
            np.random.shuffle(bg)
            frame[mask == 0] = bg
        elif self.apply_mask == 'block_black':
            frame[mask == 0] = 0
        elif self.apply_mask == 'bokeh':
            # this is the 'bokeh' value calculated as the proportion of pixels which are in the background
            frame[mask == 0] *= 0.74
        elif self.apply_mask == 'random_bg':
            # select a random frame
            random_idx = np.random.randint(0, len(self.frame_path_list))
            random_frame = np.load(self.frame_path_list[random_idx]).astype(np.uint8)
            random_mask = np.load(self.mask_path_list[random_idx]).astype(np.uint8)
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

        if self.rtn_id:
            return frame, label, id_
        else:
            return frame, label
