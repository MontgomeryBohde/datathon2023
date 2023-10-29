# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.

from data import vector_to_raster, scale, shift, rdp_simplify

import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pickle

d = {
    0: "accessory",
    1: "cats",
    2: "construction",
    3: "fruit",
    4: "instrument",
    5: "one_liner",
    6: "plant",
    7: "shape",
    8: "sport",
    9: "terrain",
    10: "tool",
    11: "vehicle",
    12: "weapon",
    13: "weather",
    14: "writing_utensil"
}

class Solution:
    def __init__(self):
        #load from pickle
        with open('models/efficientnet_v2_s_12.pkl', 'rb') as f:
            self.m1 = pickle.load(f)
            self.m1.eval()
            self.m1.cuda()    
        with open('models/convnext_tiny_12.pkl', 'rb') as f:
            self.m2 = pickle.load(f)
            self.m2.eval()
            self.m2.cuda()   
        with open('models/regnet_x_3_2gf_12.pkl', 'rb') as f:
            self.m3 = pickle.load(f)
            self.m3.eval()
            self.m3.cuda()   
        with open('models/resnext101_64x4d_12.pkl', 'rb') as f:
            self.m3 = pickle.load(f)
            self.m3.eval()
            self.m3.cuda()   
        

    # this is a signal that a new drawing is about to be sent
    def new_case(self):
        self.guesses = torch.zeros(15)
        self.listOfStrokes = []

    # given a stroke, return a string of your guess
    def guess(self, x: list[int], y: list[int]) -> str:
        self.listOfStrokes.append((x, y))
        
        data = torch.tensor(vector_to_raster([scale(shift(rdp_simplify(self.listOfStrokes)))], side=256)[0], dtype=torch.float32)
        data = (np.expand_dims(data, axis=0) - 0.12000638813804619) / 0.3186337241490653
        data = np.repeat(data, 3, axis=0)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            data = data.cuda()
            out1 = self.m1(data).cpu()
            out2 = self.m2(data).cpu()
            out3 = self.m3(data).cpu()
            out4 = self.m4(data).cpu()
            pred1 = torch.softmax(out1 - self.guesses, dim=1)
            pred2 = torch.softmax(out2 - self.guesses, dim=1)
            pred3 = torch.softmax(out3 - self.guesses, dim=1)
            pred4 = torch.softmax(out4 - self.guesses, dim=1)
            pred = torch.argmax(pred1 + pred2 + pred3 + pred4).item()
            
            self.guesses[pred] -= 1e8
            
        return d[int(pred)]
            
        

    # this function is called when you get
    def add_score(self, score: int):
        print(score)
        pass
    