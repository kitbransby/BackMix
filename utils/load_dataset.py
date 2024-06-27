from utils.dataset_tmed2_4_class import TMED2_4_Class
from utils.dataset_tmed2_4_class_semi import TMED2_4_Class_Semi
from utils.dataset_wase_4_class import WASE_4_Class
from utils.dataset_tmed2_4_class_semi_weighted import TMED2_4_Class_Semi_Weighted

from torch.utils.data import DataLoader
import albumentations as A


def load_dataset(config):

    data_root = config['DATA_ROOT']

    train_transforms = A.Compose(
        [A.Rotate(30, p=1, interpolation=2, border_mode=1),
         A.HorizontalFlip(p=0.5),
         A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1)])


    if config['DATASET'] == 'tmed2_4class':
        train_dataset = TMED2_4_Class(data_root=data_root + 'TMED2/',
                                      transforms=train_transforms,
                                      subset='train',
                                      norm=config['NORM'],
                                      rtn_id=True,
                                      apply_mask=config['APPLY_MASK'])
        val_dataset = TMED2_4_Class(data_root=data_root + 'TMED2/',
                                    transforms=None,
                                    subset='val',
                                    norm=config['NORM'],
                                    rtn_id=True,
                                    apply_mask=config['APPLY_MASK'])
        test_dataset = TMED2_4_Class(data_root=data_root + 'TMED2/',
                                     transforms=None,
                                     subset='test',
                                     norm=config['NORM'],
                                     rtn_id=True,
                                     apply_mask=config['APPLY_MASK'])
    elif config['DATASET'] == 'tmed2_4class_semi':
        train_dataset = TMED2_4_Class_Semi(data_root=data_root + 'TMED2/',
                                      transforms=train_transforms,
                                      subset='train',
                                      norm=config['NORM'],
                                      rtn_id=True,
                                      apply_mask=config['APPLY_MASK'],
                                      sup_fraction=config['SUP_FRACTION'])
        val_dataset = TMED2_4_Class_Semi(data_root=data_root + 'TMED2/',
                                    transforms=None,
                                    subset='val',
                                    norm=config['NORM'],
                                    rtn_id=True,
                                    apply_mask=config['APPLY_MASK'],
                                    sup_fraction=config['SUP_FRACTION'])
        test_dataset = TMED2_4_Class_Semi(data_root=data_root + 'TMED2/',
                                     transforms=None,
                                     subset='test',
                                     norm=config['NORM'],
                                     rtn_id=True,
                                     apply_mask=config['APPLY_MASK'],
                                     sup_fraction=config['SUP_FRACTION'])

    elif config['DATASET'] == 'tmed2_4class_semi_weighted':
        train_dataset = TMED2_4_Class_Semi_Weighted(data_root=data_root + 'TMED2/',
                                      transforms=train_transforms,
                                      subset='train',
                                      norm=config['NORM'],
                                      rtn_id=True,
                                      apply_mask=config['APPLY_MASK'],
                                      sup_fraction=config['SUP_FRACTION'],
                                      mu=config['MU']
                                                    )
        val_dataset = TMED2_4_Class_Semi_Weighted(data_root=data_root + 'TMED2/',
                                    transforms=None,
                                    subset='val',
                                    norm=config['NORM'],
                                    rtn_id=True,
                                    apply_mask=config['APPLY_MASK'],
                                    sup_fraction=config['SUP_FRACTION'],
                                    mu=config['MU']
                                                  )
        test_dataset = TMED2_4_Class_Semi_Weighted(data_root=data_root + 'TMED2/',
                                     transforms=None,
                                     subset='test',
                                     norm=config['NORM'],
                                     rtn_id=True,
                                     apply_mask=config['APPLY_MASK'],
                                     sup_fraction=config['SUP_FRACTION'],
                                     mu=config['MU']
                                                   )
    elif config['DATASET'] == 'wase_4class':
        train_dataset = WASE_4_Class(data_root=data_root + 'WASE/',
                                      transforms=train_transforms,
                                      subset='train',
                                      norm=config['NORM'],
                                      rtn_id=True,)
        val_dataset = WASE_4_Class(data_root=data_root + 'WASE/',
                                    transforms=None,
                                    subset='val',
                                    norm=config['NORM'],
                                    rtn_id=True,)
        test_dataset = WASE_4_Class(data_root=data_root + 'WASE/',
                                     transforms=None,
                                     subset='test',
                                     norm=config['NORM'],
                                     rtn_id=True,)

    else:
        print('WARNING - No dataset selected..')

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                              num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], num_workers=config['NUM_WORKERS'],
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], num_workers=config['NUM_WORKERS'],
                             pin_memory=True)

    return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)