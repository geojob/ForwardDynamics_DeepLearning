import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
np.set_printoptions(suppress=True)


class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Y': self.Y[idx]}


class Net(nn.Module):
    # ---
    # Your code goes here
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim,54)
        self.fc2 = nn.Linear(54,12)
        self.fc4 = nn.Linear(12,6)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        return x
        
    def predict(self,feat):
        self.eval()
        myfeat = torch.from_numpy(feat).T.float()
        res = self.forward(myfeat).detach().numpy()
        return res.T
    
 
        
    # ---
class TrainNet(object):
    def __init__(self, network):
        self.network = network
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learning_rate)
        self.loss = nn.MSELoss()
        self.num_epochs = 950
      
    def train(self, trainloader, testloader,args):
        self.network.train()
        for e in range(1,self.num_epochs+1):
            totalloss = 0.0
            for k, data in enumerate(trainloader,0):
                 state = data['X'].float()
                 newstate = data['Y'].float()
                 self.optimizer.zero_grad()
                 predstate = self.network(state)
                 loss = self.loss(predstate,newstate)
                 loss.backward()
                 totalloss+=loss.item()
                 self.optimizer.step()
            test_loss = self.test(testloader)
            model_folder_name = f'epoch3_{e:04d}_loss_{test_loss:.8f}'
            if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
               os.makedirs(os.path.join(args.save_dir, model_folder_name))
            torch.save(self.network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
            print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')
    
    
    
    def test(self,dataloader):
         self.network.eval()
         test_loss = 0.0
         for z, data in enumerate(dataloader,0):
            testnew = self.network(data['X'].float())
            actual = data['Y'].float()
            loss = self.loss(testnew,actual)
            test_loss+=loss.item()
         return test_loss/z


        
        



def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=256)

    # ---
    # Your code goes here
    mynet = Net(9)
    trainproc = TrainNet(mynet)
    trainproc.train(train_loader,test_loader,args)
    
    # ---


if __name__ == '__main__':
    main()
