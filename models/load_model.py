#from learnt_focus.models.inceptionv3 import InceptionV3
from learnt_focus.models.resnet import ResNet18, ResNet34
#from learnt_focus.models.resnet_SMA import ResNet_SMA


def load_model(config, device):
    # if config['MODEL'] == 'inception':
    #     model = InceptionV3(config['NUM_CLASSES']).to(device)
    if config['MODEL'] == 'resnet18':
        model = ResNet18(config['INP_DIM'], config['NUM_CLASSES']).to(device)
    elif config['MODEL'] == 'resnet34':
        model = ResNet34(config['INP_DIM'], config['NUM_CLASSES']).to(device)
    #elif config['MODEL'] == 'resnet18_SMA':
    #    model = ResNet_SMA(config['INP_DIM'], config['NUM_CLASSES']).to(device)

    return model