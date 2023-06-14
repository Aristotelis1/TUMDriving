import sys
sys.path.append('.')

import os
import argparse

import torch
from networks.resnet import resnet_model
from torch.utils.data import DataLoader
from data_preparation.resnet_data import SimulatorDataset


def train_val(cfg):
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('gpu:0')

    train_data = SimulatorDataset(data_path = cfg.train_data)
    val_data = SimulatorDataset(data_path = cfg.val_data)

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = resnet_model().to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_function = torch.nn.MSELoss()
    
    for epoch_i in range(cfg.epochs):
        model.train()

        for train_i, (input, target) in enumerate(train_dataloader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = loss_function(output, target)

            print(loss.item())

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_sum = 0
            for val_i, (input, target) in enumerate(val_dataloader):
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss_sum = loss_sum + loss_function(output, target)
            print(val_i)
            print('val_loss:', loss_sum.item()/(val_i+1))

        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--train-data', type=str, default='data/simulator/train/Dataset22/VehicleData.txt', help='data path')
    parser.add_argument('--val-data', type=str, default='data/simulator/train/Dataset22/VehicleData.txt', help='data path')
    parser.add_argument('--device', type=str, default='cpu', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet', help='path to save checkpoint')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    
    train_val(cfg)

    