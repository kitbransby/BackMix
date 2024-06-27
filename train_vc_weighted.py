import argparse
import os
import yaml
import datetime
import time
import pickle
import sys
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from models.load_model import load_model

from utils.train_utils import plot
from utils.load_dataset import load_dataset

def main(config):

    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    model = load_model(config, device)

    summary(model, input_size=(config['BATCH_SIZE'], config['INP_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    loss_function = nn.CrossEntropyLoss(reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), config['LR'])
    max_epochs = config['EPOCHS']

    save_folder = os.path.join("results", config['RUN_ID'] + '_' + config['DATASET'])
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print('Warning: ', e)

    best_accuracy = 0
    best_loss = np.inf
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    print('Starting Training...')
    for epoch in range(max_epochs):
        start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        correct_pred = 0
        incorrect_pred = 0
        for X, Y, loss_weights, id_ in train_loader:
            X, Y, loss_weights = X.to(device), Y.to(device), loss_weights.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)
            if config['MODEL'] == 'inception':
                Y_hat = Y_hat[0]
            loss = loss_function(Y_hat, Y)
            loss = loss * loss_weights
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct_pred += (Y_hat.argmax(dim=1) == Y).sum()
            incorrect_pred += (Y_hat.argmax(dim=1) != Y).sum()
            if config['VERBOSE']:
                if step % config['TRAIN_PRINT'] == 0:
                    print(f"{step}/{len(train_dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.5f}")
            step += 1
            #break
        epoch_loss /= step
        train_loss_all.append(epoch_loss)
        accuracy = correct_pred / (correct_pred + incorrect_pred)
        train_acc_all.append(accuracy.cpu().numpy())
        print(f"Train epoch: {epoch + 1} avg loss: {epoch_loss:.4f}, avg acc: {accuracy:.2f}" )


        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            step = 0
            correct_pred = 0
            incorrect_pred = 0
            for X, Y, loss_weights, id_ in val_loader:
                X, Y, loss_weights = X.to(device),Y.to(device), loss_weights.to(device)
                Y_hat = model(X)
                loss = loss_function(Y_hat, Y)
                loss = loss * loss_weights
                loss = loss.mean()

                epoch_loss += loss.item()
                correct_pred += (Y_hat.argmax(dim=1) == Y).sum()
                incorrect_pred += (Y_hat.argmax(dim=1) != Y).sum()
                if config['VERBOSE']:
                    if step % config['VAL_PRINT'] == 0:
                        print(f"{step}/{len(val_dataset) // val_loader.batch_size}, " f"val_loss: {loss.item():.5f}")
                step += 1
                #break
            epoch_loss /= step
            val_loss_all.append(epoch_loss)
            accuracy = correct_pred / (correct_pred + incorrect_pred)
            val_acc_all.append(accuracy.cpu().numpy())
            print(f"Val epoch: {epoch + 1} avg loss: {epoch_loss:.4f}, avg acc: {accuracy:.2f}" )
            end = time.time()
            epoch_time = end - start
            print('Epoch time: {:.2f}s'.format(epoch_time))
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_acc.pt'))
                print("saved model new best acc")
            if epoch_loss <= best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_loss.pt'))
                print("saved model new best loss")


        #print('Training done, saving logs to {}'.format(save_folder))
        with open(save_folder+'/logs.pkl', 'wb') as f:
            pickle.dump([train_loss_all, val_loss_all,train_acc_all, val_acc_all], f)

        plot(train_loss_all, val_loss_all,train_acc_all, val_acc_all, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str)
    parser.add_argument('--LOAD_TO_RAM', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--CONFIG', type=str)
    parser.add_argument('--SEED', type=int)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    config['RUN_ID'] = datetime.datetime.now().strftime('%m_%d_%H_%M_%S.%f') + '_' + config['MODEL']

    print('config: ', config)

    main(config)
