#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install torchdata


# In[2]:


# !wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
# !unzip ml-1m.zip


# In[3]:


from torchdata import datapipes as dp

# In[4]:


import torch
from torch import nn, div, square, norm
from torch.nn import functional as F
from torchdata import datapipes as dp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

datapath = 'ml-1m/'
seed = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_users = pd.read_csv(datapath + 'users.dat',
                        delimiter='::',
                        engine='python',
                        encoding='latin-1',
                        header=None)[0].max()
num_items = pd.read_csv(datapath + 'movies.dat',
                        delimiter='::',
                        engine='python',
                        encoding='latin-1',
                        header=None)[0].max()

num_users, num_items

train_items, test_items = train_test_split(torch.arange(num_items),
                                           test_size=0.2,
                                           random_state=seed)
train_items.size(), test_items.size()

# create global user_item matrix and mask matrix
user_item_mat = torch.zeros((num_users, num_items))

ratings = pd.read_csv(datapath + 'ratings.dat',
                      encoding='latin-1',
                      header=None,
                      engine='python',
                      delimiter='::')


def create_data_from_line(line):
    user_id, item_id, rating, *_ = line
    user_item_mat[user_id - 1, item_id - 1] = rating
    return None


ratings.T.apply(create_data_from_line)

torch.where(user_item_mat == 0, 1, 0).sum() / (num_users * num_items)


def collate_fn(batch):
    return torch.LongTensor(batch)


def create_datapipe_from_array(array, mode='train', batch_size=32, len=1000):
    pipes = dp.iter.IterableWrapper(array)
    pipes = pipes.shuffle(buffer_size=len)
    pipes = pipes.sharding_filter()

    if mode == 'train':
        pipes = pipes.batch(batch_size, drop_last=True)
    else:
        pipes = pipes.batch(batch_size)

    pipes = pipes.map(collate_fn)
    return pipes


batch_size = 512

train_dp = create_datapipe_from_array(train_items, batch_size=batch_size)
test_dp = create_datapipe_from_array(test_items, mode='test', batch_size=batch_size)

num_workers = 2

train_dl = DataLoader(dataset=train_dp, shuffle=True)
test_dl = DataLoader(dataset=test_dp, shuffle=False)


class AutoRec(nn.Module):
    def __init__(self, d, k, lambda_):
        super().__init__()
        self.lambda_ = lambda_
        self.W = nn.Parameter(torch.randn(d, k))
        self.V = nn.Parameter(torch.randn(k, d))
        self.mu = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.randn(d))

    def regularization(self):
        return div(self.lambda_, 2) * (square(norm(self.W)) + square(norm(self.V)))

    def forward(self, r):
        encoder = self.V.matmul(r.T).T + self.mu
        return self.W.matmul(encoder.sigmoid().T).T + self.b


def train_epoch(model, dl, opt, criterion):
    list_loss = []
    start_time = time.perf_counter()
    for batch_idx, items_idx in enumerate(dl):
        r = user_item_mat[:, items_idx].squeeze().permute(1, 0).to(device)
        r_hat = model(r)
        loss = criterion(r, r_hat * torch.sign(r)) + model.regularization()

        list_loss.append(loss.item())
        if batch_idx % 50 == 0:
            log_time = round(time.perf_counter() - start_time, 4)
            print("Loss {:.2f} | {:.4f}s".format(loss.item(), log_time))

        opt.zero_grad()
        loss.backward()
        opt.step()

    return list_loss


def eval_epoch(model, dl, criterion):
    model.eval()
    truth = []
    predict = []
    list_loss = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, items_idx in enumerate(dl):
            r = user_item_mat[:, items_idx].squeeze().permute(1, 0).to(device)

            r_hat = model(r)

            truth.append(r)
            predict.append(r_hat * torch.sign(r))

            loss = criterion(r, r_hat * torch.sign(r)) + model.regularization()

            list_loss.append(loss.item())
            if batch_idx % 30 == 0:
                log_time = round(time.perf_counter() - start_time, 4)
                print("Loss {:.2f} | {:.4f}s".format(loss.item(), log_time))

    rmse = torch.Tensor([torch.sqrt(square(r - r_hat).sum() / torch.sign(r).sum())
                         for r, r_hat in zip(truth, predict)]).mean().item()

    return list_loss, rmse


model = AutoRec(d=num_users, k=500, lambda_=0.0001).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.012, weight_decay=1e-5)
criterion = nn.MSELoss().to(device)


max_epochs = 100
losses = []
val_losses = []


for epoch in range(max_epochs):
    print("=" * 10 + f"Epoch: {epoch}" + "=" * 10)
    epoch_loss = train_epoch(model, train_dl, opt, criterion)
    val_loss, rmse = eval_epoch(model, test_dl, criterion)
    losses.extend(epoch_loss)
    val_losses.extend(val_loss)


plt.plot(losses)
plt.plot(val_losses)
plt.tight_layout()
plt.show()


val_loss, rmse = eval_epoch(model, test_dl, criterion)
rmse

