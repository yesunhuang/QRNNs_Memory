'''
Name: Helpers
Desriptption: Some helper functions
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: for studying quantum RNN
Author: YesunHuang
Date: 2022-04-02 23:10:18
'''

#import everything
from IPython import display
import time
import matplotlib.pyplot as plt
import numpy as np

class Animator:  
    """drawing data with animation"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        '''
        name: __init__
        fuction: initialize the animator
        param {xlabel}: the label of x
        param {ylabel}: the label of y
        param {legend}: the legend
        param {xlim}: the limit of x
        param {ylim}: the limit of y
        param {xscale}: the scale of x
        param {yscale}: the scale of y
        param {fmts}: the format of lines
        param {nrows}: the number of rows
        param {ncols}: the number of columns
        param {figsize}: the size of figure
        '''        
        # draw grids
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # define paramters catcher
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def set_axes(self,axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        '''
        name: set_axes
        fuction: set the axes
        param {axes}: the axes
        param {xlabel}: the label of x
        param {ylabel}: the label of y
        param {xlim}: the limit of x
        param {ylim}: the limit of y
        param {xscale}: the scale of x
        param {yscale}: the scale of y
        param {legend}: the legend
        '''        
        # set label
        if xlabel is not None:
            axes.set_xlabel(xlabel)
        if ylabel is not None:
            axes.set_ylabel(ylabel)
        # set scale
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        # set limit
        if xlim is not None:
            axes.set_xlim(*xlim)
        if ylim is not None:
            axes.set_ylim(*ylim)
        # set legend
        if legend is not None:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        '''
        name: add
        fuction: add data
        param {x}: the x data
        param {y}: the y data
        '''        
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class Timer:
    """Record multiple running times."""

    def __init__(self):
        '''
        name: __init__ 
        fuction: initialize the timer
        '''    
        self.times = []
        self.start()

    def start(self):
        '''
        name: start
        fuction: start the timer
        '''        
        self.tik = time.time()

    def stop(self):
        '''
        name: stop
        fuction: stop the timer and record the time in a list
        '''        
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        '''
        name: avg
        fuction: return the average time
        '''        
        return sum(self.times) / len(self.times)

    def sum(self):
        '''
        name: sum
        fuction: return the sum of time
        '''        
        return sum(self.times)

    def cumsum(self):
        '''
        name: cumsum
        fuction: Return the accumulated time
        '''        
        return np.array(self.times).cumsum().tolist()

class Accumulator:  
    """accumulate on n variables"""

    def __init__(self, n):
        '''
        name: __init__
        fuction: initialize the accumulator
        '''        
        self.data = [0.0] * n

    def add(self, *args):
        '''
        name: add
        fuction: add the data
        param {*args}: data to be added
        '''        
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        '''
        name: reset
        fuction: reset the data
        '''        
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        '''
        name: __getitem__
        fuction: get the data
        param {idx}: the index of data
        '''        
        return self.data[idx]