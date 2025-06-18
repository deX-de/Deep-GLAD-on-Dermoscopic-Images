# Raising the Bar in Graph-level Anomaly Detection (GLAD)
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

import time
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from utils import compute_pre_recall_f1, format_time
import copy
from .base_trainer import BaseTrainer

class OCGTL_Trainer(BaseTrainer):
    def __init__(self, model, loss_function, device='cuda', semi=False):
        super().__init__(model, device)
        self.model = self.model.to(self.device)
        self.loss_fun = loss_function
        self.semi = semi

    def train(self, train_loader, optimizer, **kwargs):
        self.model.train()
        loss_all = 0
        num_sample = 0
        for data in train_loader:
            data = data.to(self.device)
            z = self.model(data)
            loss = self.compute_loss(z, data)
            loss_mean = loss.mean()

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss_mean.item() * data.num_graphs
            num_sample += data.num_graphs

        return loss_all / num_sample

    def evaluate(self, data_loader, normal_class=None):
        self.model.eval()
        score_list, label_list = [], []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                z = self.model(data)
                score = self.compute_loss(z, data, eval=True)
                label_list.append(data.y)
                score_list.append(score)

        score_list = torch.cat(score_list).cpu().numpy()
        label_list = torch.cat(label_list).cpu().numpy()
        
        binary_labels = (label_list != normal_class).astype(int)

        roc_auc = roc_auc_score(binary_labels, score_list)
        pr_auc = average_precision_score(binary_labels, score_list)
        return roc_auc, pr_auc
    
    def run_training(self, train_loader, optimizer, num_epochs, validation_loader=None, 
                     test_loader=None, normal_class=None, scheduler=None, early_stopper=None, 
                     logger=None, log_every=1):
        self.init_center(train_loader)
        return super().run_training(train_loader, optimizer, num_epochs, validation_loader, test_loader, normal_class, scheduler, early_stopper, logger, log_every)

    def compute_loss(self, z, data, eval=False):
        if self.semi:
            return self.loss_fun(z, data.y, eval=eval)
        else:
            return self.loss_fun(z, eval=eval)


    def init_center(self, train_loader):
        try:
            self.model.eval()
            self.model.init_center(train_loader)
        except AttributeError:
            pass
    