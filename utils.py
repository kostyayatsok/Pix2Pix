from collections import defaultdict
from typing import List
import time
import torch

class MetricTracker:
    def __init__(self, eternals: List=[]) -> None:
        self.value = defaultdict(int)
        self.count = defaultdict(int)
        self.eternals = eternals

    def __call__(self, batch, count=1, suffix='', exclude=[]) -> None:
        for key, value in batch:
            if key in exclude:
                continue
            if isinstance(value, int) or isinstance(value, float):
                pass
            elif isinstance(value, torch.Tensor) and len(value.size()) == 0:
                value = value.item()
            else:
                continue
            self.value[key+'_'+suffix] += value
            self.count[key+'_'+suffix] += count
                
    def __getitem__(self, key, suffix='') -> float:
        if key not in self.count or self.count == 0:
            return None
        res = self.value[key+'_'+suffix] / self.count[key+'_'+suffix]
        if not key.emdswith(self.eternals):
            self.value[key+'_'+suffix] = 0
            self.count[key+'_'+suffix] = 0
        return res

    def all(self):
        res = {}
        for key in self.count:
            avg = self.__getitem__(key)
            if avg is not None:
                res[key] = avg
        return res
    
    

class Timer:
    def __init__(self) -> None:
        self.start_time = defaultdict(int)
        self.end_time = defaultdict(int)
    def start(self, name: str='default') -> None:
        start_time = time.perf_counter()
        self.start_time[name] = start_time
    def end(self, name: str='default') -> None:
        self.end_time[name] = time.perf_counter()
    def get(self, name: str='default') -> float:
        return self.end_time[name] - self.start_time[name]
    def all(self):
        res = {}
        for name in self.end_time:
            avg = self.get(name)
            if avg is not None:
                res[name] = avg
        return res