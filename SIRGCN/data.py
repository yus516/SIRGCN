import sys
import torch
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable

import pandas as pd
from datetime import datetime

# Define a function to determine the season based on the month
def get_season(month):
    
    # with 20 weeks shift, around 5 months earlier. 
    if 10 <= month <= 12:
        return "Spring"
    elif 1 <= month <= 3:
        return "Summer"
    elif 4 <= month <= 6:
        return "Fall"
    else:
        return "Winter"

def get_season_index(start_date, end_date, data_size, seq_len):

    print("start: " + str(start_date) + ", end: " + str(end_date))

    # Create a list of dates from start to end, incremented by one week
    dates = pd.date_range(start_date, end_date, freq='W')

    # Create a DataFrame with the dates as the index
    df = pd.DataFrame(index=dates)

    # Create a new column in the DataFrame with the season for each date
    df['month'] = df.index.month
    df['season'] = df['month'].apply(get_season)

    df = df[0:data_size-seq_len]
    df.index = pd.RangeIndex(len(df))
    spring_idx = df[df['season'] == 'Spring'].index.values
    summer_idx = df[df['season'] == 'Summer'].index.values
    fall_idx = df[df['season'] == 'Fall'].index.values
    winter_idx = df[df['season'] == 'Winter'].index.values

    return spring_idx+seq_len, summer_idx+seq_len, fall_idx+seq_len, winter_idx+seq_len

class DataBasicLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.d = 0
        self.add_his_day = False
        self.save_dir = args.save_dir
        self.rawdat = np.loadtxt(open("data/{}.txt".format(args.dataset)), delimiter=',')
        print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)

        if args.extra:
            self.load_external(args)
 
        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # n_sample, n_group
        print(self.n, self.m)

        self.scale = np.ones(self.m)

        self._pre_train_seasonal(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('size of train/val/test sets',len(self.train[0]),len(self.val[0]),len(self.test[0]))
    
    def load_label_file(self, filename):
        labelfile = pd.read_csv("data/"+filename+".csv", header=None)
        labelLen = len(labelfile)
        label = dict()
        for i in range(labelLen):
            label[labelfile.iloc[i,0]]=labelfile.iloc[i,1]
        return label, labelLen

    def load_external(self, args):
        label, label_num = self.load_label_file(args.label)
        files = os.listdir("data/{}".format(args.extra))
        filesLen = len(files)
        extra_adj_list = []
        for i in range(filesLen):
            snapshot = pd.read_csv("data/"+args.extra+"/"+files[i], header=None)
            snapshot_len = len(snapshot)
            extra_adj = np.zeros((label_num,label_num))
            for j in range(snapshot_len):
                extra_adj[label[snapshot.iloc[j,0]],label[snapshot.iloc[j,1]]] = snapshot.iloc[j,2]
            #print(extra_adj)
            extra_adj_list.append(extra_adj)
        extra_adj = torch.Tensor(np.array(extra_adj_list))
        print('external information', extra_adj.shape)
        self.external = Variable(extra_adj)
        if args.cuda:
            self.external = extra_adj.cuda()

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(open("data/{}.txt".format(args.sim_mat)), delimiter=','))
        self.orig_adj = self.adj
        self.degree_adj = torch.sum(self.orig_adj, dim=-1)
        #rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        #self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()
            self.orig_adj = self.orig_adj.cuda()
            self.degree_adj = self.degree_adj.cuda()

    def _pre_train(self, train, valid, test):
        self.train_set = train_set = range(self.P+self.h-1, train)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() #199, 47
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        print(self.dat.shape)
        print(len(self.train_set))
        print(len(self.valid_set))
        print(len(self.test_set))

    def _pre_train_seasonal(self, train, valid, test):
        Spring, Summer, Fall, Winter = get_season_index((2012, 8, 1), (2020, 3, 31), 348, 20)
        # Spring, Summer, Fall, Winter = get_season_index((2012, 8, 1), (2020, 3, 31), 360, 20)
        train_season = np.concatenate((Winter, Summer), axis=0)
        test_season = np.concatenate((Spring, Fall), axis=0)
        thresh = int(len(train_season) * 0.2)
        self.train_set = train_set = train_season[:-thresh]
        self.valid_set = valid_set = train_season[-thresh:]
        self.test_set = test_set = test_season
        self.full_set = np.array(range(20, 348))
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() #199, 47
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        # self.population = torch.tensor(np.sum(self.dat[10:30, :], axis=0), dtype=torch.float32)
        self.population = torch.tensor(np.sum(self.dat, axis=0), dtype=torch.float32) / 10
        print(self.dat.shape)
        print(len(self.train_set))
        print(len(self.valid_set))
        print(len(self.test_set))
         
    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        self.full = self._batchify(self.full_set, self.h)
        if (train == valid):
            self.val = self.test
 
    def _batchify(self, idx_set, horizon, useraw=False): 

        n = len(idx_set)
        Y = torch.zeros((n, self.m))
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P+1, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            if useraw: # for narmalization
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.rawdat[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51 : # at least 52
                        his_day = self.dat[idx_set[i]-52:idx_set[i]-51, :] #
                    else: # no history day data
                        his_day = np.zeros((1,self.m))

                    his_window = np.concatenate([his_day,his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i,:self.P+1,:] = torch.from_numpy(his_window) # size (window+1, m)
                else:
                    X[i,:self.P,:] = torch.from_numpy(his_window) # size (window, m)
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    # original
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)
            #print('x shape', X.shape) # batch_size window_size region_num
            #print('y shape', Y.shape)

            data = [model_inputs, Variable(Y), index]
            yield data
            start_idx += batch_size

   