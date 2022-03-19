'''
Name: GradFreeOptimizer
Desriptption: The base class for all gradient free optimizer
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For quantum NNs
Author: YesunHuang
Date: 2022-03-20 00:42:27
'''

### import everything
import abc
import torch

class StandardGradFreeOptimizer(metaclass=abc.ABCMeta):
    '''A abstract base class for standard gradient free optimizers'''
    
    @abc.abstractmethod
    @torch.no_grad()
    def step(self,**kwarg):
        '''A abstract method which update the weight'''
        pass