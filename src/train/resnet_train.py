import sys
sys.path.append('.')

import os
import argparse

import torch
from networks.resnet import resnet_model
from torch.utils.data import DataLoader, random_split
from data_preparation.resnet_data import SimulatorDataset


def train_val(cfg):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    # if cfg.device == 'cpu':
    #     device = torch.device('cpu')
    # else:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
    #     device = torch.device('gpu:0')

    dataset = SimulatorDataset(data_path = cfg.data)
    # val_data = SimulatorDataset(data_path = cfg.val_data)
    
    # Define the lengths for the split datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    print("Training dataset size:", len(train_data))
    print("Validation dataset size:", len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = resnet_model().to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_function = torch.nn.MSELoss()
    
    for epoch_i in range(cfg.epochs):
        model.train()
        loss = 0.0
        for train_i, (input, target) in enumerate(train_dataloader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()
            input = input.type(torch.float32)
            output = model(input)
            loss = loss_function(output, target)

            loss += loss.item()

            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            for val_i, (input, target) in enumerate(val_dataloader):
                input, target = input.to(device), target.to(device)
                input = input.type(torch.float32)
                output = model(input)
                loss_sum = loss_sum + loss_function(output, target)
        print(f"Epoch: {epoch_i+1}, train_loss: {loss/(train_i+1)} val_loss: {loss_sum.item()/(val_i+1)}")
        torch.save(model, cfg.save_path+'/'+ 'resnet_' + str(epoch_i + 1)+'.pt')
        torch.cuda.empty_cache()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='total number of training epochs')
    parser.add_argument('--data', type=str, default='data/simulator/train/Dataset2/VehicleData.txt', help='data path')
    # parser.add_argument('--val-data', type=str, default='data/simulator/train/Dataset2/VehicleData.txt', help='data path')
    parser.add_argument('--device', type=str, default='cpu', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.001, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet', help='path to save checkpoint')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    
    train_val(cfg)

    