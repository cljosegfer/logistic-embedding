#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:56:25 2022

@author: jose
"""

import torch.nn as nn

class ae(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(kwargs["n"], 128), 
            nn.ReLU(inplace = True), 
            nn.Linear(128, 128), 
            nn.ReLU(inplace = True))
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 128), 
            nn.ReLU(inplace = True), 
            nn.Linear(128, kwargs["n"]))
    
    def forward(self, X):
        H = self.encoder(X)
        X_hat = self.decoder(H)
        return X_hat
