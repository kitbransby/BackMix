import argparse
import os
import yaml
import numpy as np
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.load_dataset import load_dataset
from models.load_model import load_model

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


def main(config):

    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    model = load_model(config, device)
    model.load_state_dict(torch.load(os.path.join('results', config['RUN_ID'], 'best_acc.pt')))

    if config['MASK_TEST_SET']:
        masking_str = '_masked_test'
    else:
        masking_str = ''

    save_folder = os.path.join("results", config['RUN_ID'], 'evaluation_' + config['DATASET'] + masking_str)
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print('Warning: ', e)


    Y_pred_all = np.zeros(len(test_dataset), dtype=np.int32)
    Y_true_all = np.zeros(len(test_dataset), dtype=np.int32)
    Y_softmax_all = np.zeros((len(test_dataset), config['NUM_CLASSES']), dtype=np.float32)
    avg_time = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):

            example = test_dataset[i]
            if len(example) == 4:
                X, X_masked, Y, id_ = example
                X = X.unsqueeze(0).to(device)
                X_masked = X_masked.unsqueeze(0).to(device)
                start = time.time()
                output = model(X, X_masked)
                end = time.time()
                avg_time.append(end - start)

            else:
                X, Y, id_ = example
                X = X.unsqueeze(0).to(device)
                start = time.time()
                output = model(X)
                end = time.time()
                avg_time.append(end - start)


            if len(output) == 2:
                Y_logits, _ = output
            else:
                Y_logits = output

            Y_softmax = F.softmax(Y_logits, dim=1).cpu().numpy()

            Y_pred = np.argmax(Y_softmax, axis=1)[0]
            if Y.numel() > 1:
                Y = torch.argmax(Y)
            Y = Y.numpy()

            #print(Y_pred.shape, Y.shape)
            #print(Y_pred, Y, Y_softmax)

            Y_pred_all[i] = Y_pred
            Y_true_all[i] = Y
            Y_softmax_all[i,:] = Y_softmax

    # evaluation
    avg_time = np.mean(avg_time)
    print('Avg inference time: {:.4f}'.format(avg_time))
    cm = confusion_matrix(Y_true_all, Y_pred_all, labels=list(range(config['NUM_CLASSES'])))

    cm_display = ConfusionMatrixDisplay.from_predictions(Y_true_all, Y_pred_all,
                                                         display_labels=test_dataset.avc_classes,
                                                         xticks_rotation="vertical",
                                                         normalize='true')
    fig, ax = plt.subplots(figsize=(20, 20))
    cm_display.plot(ax=ax)
    plt.tight_layout()
    plt.savefig(save_folder + '/confusion_matrix.png', dpi=200, pad_inches=5)
    plt.close('all')

    clf_report = classification_report(Y_true_all, Y_pred_all, labels=list(range(config['NUM_CLASSES'])),
                                       target_names=test_dataset.avc_classes, digits=4, output_dict=False)
    print(clf_report)
    results = {'clf_report': clf_report, 'confusion_matrix': cm, 'speed': avg_time}

    print('Saving predictions and scores to {}'.format(save_folder))
    with open(save_folder + '/scores.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(save_folder+'/predictions.pkl', 'wb') as f:
        pickle.dump([Y_true_all, Y_pred_all, Y_softmax], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str)
    parser.add_argument('--RUN_ID', type=str)
    parser.add_argument('--CONFIG', type=str)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)
