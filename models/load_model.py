from models.resnet import ResNet18, ResNet34

def load_model(config, device):
    if config['MODEL'] == 'resnet18':
        model = ResNet18(config['INP_DIM'], config['NUM_CLASSES']).to(device)
    elif config['MODEL'] == 'resnet34':
        model = ResNet34(config['INP_DIM'], config['NUM_CLASSES']).to(device)
    return model