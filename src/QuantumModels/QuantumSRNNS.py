'''
Name: QuantumSRNNs
Desriptption: Implement the classical simple recurrent neural networks
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For studying quantum recurrrent neural networks
Author: YesunHuang
Date: 2022-04-06 10:18:00
'''

#import everything
import torch
from torch import nn
from collections.abc import Callable
import qutip as qt
#modify path
from GradientFreeOptimizers.CostFunc import StandardSNN
import GradientFreeOptimizers.Helpers as hp

class QuantumSRNN(StandardSNN):
    '''implement the quantum sRNN'''

    def __init__(self, inputSize:int=1, qubits:int=10, outputSize:int=1,\
                get_params:Callable=None,init_state:Callable=None,forward_fn:Callable=None):
        '''
        name:__init__
        function: initialize the class for quantum SRNN
        param {inputSize}: the size of input
        param {qubits}: the number of qubits
        param {outputSize}: the size of output
        param {get_params}: the function to get the parameters
        param {init_state}: the function to initialize the state
        param {forward_fn}: the function to forward the quantum network
        '''
        self.inputSize,self.qubits,self.outputSize = inputSize,qubits,outputSize
        (self.params,self.constants)=get_params(inputSize,qubits,outputSize)
        self.init_state,self.forward_fn=init_state,forward_fn

    def __call__(self,X:torch.Tensor,state:tuple):
        '''
        name:__call__
        function: call the quantum SRNN
        param {X}: the input
        param {state}: the state
        return: the output
        '''
        return self.forward_fn(X,state,(self.params,self.constants))
    
    def call_with_weight(self, X: torch.Tensor, weight: tuple):
        '''
        name:call_with_weight
        function: call the quantum SRNN with weight
        param {X}: the input
        param {weight}: the weight
        return: the output
        '''
        return self.forward_fn(X.transpose(0,1),self.begin_state(X.shape[0]),(weight,self.constants))

    def begin_state(self,batch_size:int=1):
        '''
        name:begin_state
        function: begin the state
        param {batch_size}: the batch size
        return: the state
        '''
        return self.init_state(batch_size,self.qubits)
