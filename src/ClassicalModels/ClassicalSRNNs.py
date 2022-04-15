'''
Name: ClassicalSRNNs
Desriptption: Implement the classical simple recurrent neural networks
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For studying quantum recurrrent neural networks
Author: YesunHuang
Date: 2022-03-28 21:49:20
'''

#import everything
from typing import Tuple
import torch
from torch import nn
from collections.abc import Callable
#modify path
from GradientFreeOptimizers.CostFunc import StandardSNN
import GradientFreeOptimizers.Helpers as hp

class ClassicalSRNN(StandardSNN):
    '''implement the classical sRNN'''

    def __init__(self, inputSize:int=1, hiddenSize:int=10, outputSize:int=1,\
                get_params:Callable=None,init_state:Callable=None, forward_fn:Callable=None):
        '''
        name: __init__ 
        function: initialize the class for classical SRNN
        param {inputSize}: the size of input
        param {hiddenSize}: the size of hidden units
        param {outputSize}: the size of output
        param {getParams}: the function to get parameters
        param {initState}: the function to initialize the state
        param {forwardFn}: the function to forward
        '''
        self.inputSize,self.hiddenSize,self.outputSize = inputSize,hiddenSize,outputSize
        (self.params,self.constants)=get_params(inputSize,hiddenSize,outputSize)
        self.init_state,self.forward_fn = init_state,forward_fn
    
    def __call__(self,X:torch.Tensor,state:tuple):
        '''
        name: __call__
        function: call the class
        param {input}: the input
        param {state}: the state
        return: the output, the new state
        '''
        return self.forward_fn(X.transpose(0,1),state,(self.params,self.constants))

    def call_with_weight(self, X:torch.Tensor, weight:tuple):
        '''
        name: call_with_weight
        function: call the class with weight
        param {X}: the input
        param {weight}: the params and state weight:(self.params)
        return: the output
        '''
        Y,_=self.forward_fn(X.transpose(0,1),\
            self.begin_state(X.shape[0]),(weight,self.constants))
        Y=Y.reshape((-1,X.shape[0],Y.shape[-1])).transpose(0,1)
        return Y
    
    def begin_state(self,batch_size:int=1):
        '''
        name: begin_state
        function: get the begin state
        param {batch_size}: the batch size
        return: the begin state
        '''
        return self.init_state(batch_size,self.hiddenSize)

class SuportFunction:
    '''Some support function for the classical sRNN'''

    def __init__(self):
        '''
        name: __init__
        function: initialize the class
        '''

    def get_init_state_fun(self,initStateValue:float=0.0):
        '''
        name: get_init_state_fun
        function: get the function to initialize the state
        param {initStateValue}: the value of the state
        return: the function
        '''
        self.initStateValue=initStateValue

        def init_state(batch_size:int,hiddenSize:int):
            '''
            name: init_state
            function: initialize the state
            param {batch_size}: the batch size
            param {hiddenSize}: the hidden size
            return: the state
            '''
            self.batch_size,self.hiddenSize=batch_size,hiddenSize
            return (torch.full((batch_size,hiddenSize),self.initStateValue),)
        return init_state

    def get_get_params_fun(self,inputRatio:float=1.0, outputRatio:float=1.0,\
                            rescale:float=0.01,inactive:list=[]):
        '''
        name: get_get_params_fun
        function: get the function to get the parameters
        param {inputRatio}: the ratio of input units
        param {outputRatio}: the ratio of output units
        param {inActive}: the params set to be inactive
                'WeightInput', 'DeltaInput', 'J', 'WeightOutput', 'DeltaOutput'
        return: the function
        '''
        self.rescale=rescale
        self.inactive=inactive
        
        def normal(shape):
            '''
            name: normal
            function: get the normal distribution
            param {shape}: the shape of the distribution
            param {rescale}: the rescale of the distribution
            return: the distribution
            '''
            return torch.randn(size=shape)*rescale

        def get_params(inputSize:int,hiddenSize:int,outputSize:int):
            '''
            name: get_params
            function: get the parameters
            param {inputSize}: the size of input
            param {hiddenSize}: the size of hidden units
            param {outputSize}: the size of output
            return: the parameters
            '''
            self.inputSize,self.hiddenSize,self.outputSize=inputSize,hiddenSize,outputSize
            self.inputUnits,self.outputUnits=int(hiddenSize*inputRatio),int(hiddenSize*outputRatio)
            params=[]
            constants=[]
            #Input params
            if 'WeightInput' in inactive:
                WInParam=rescale*torch.ones((inputSize,self.inputUnits)).detach_()
                constants.append(WInParam)
            else:
                WInParam=normal((inputSize,self.inputUnits)).requires_grad_(True)
                params.append(WInParam)
            WInZeroPad=torch.zeros((inputSize,self.hiddenSize-self.inputUnits)).detach_()
            constants.append(WInZeroPad)
            if 'DeltaInput' in inactive:
                DeltaInParam=torch.zeros(self.inputUnits).detach_()
                constants.append(DeltaInParam)
            else:
                DeltaInParam=torch.zeros(self.inputUnits).requires_grad_(True)
                params.append(DeltaInParam)
            DeltaInPad=torch.zeros(self.hiddenSize-self.inputUnits).detach_()
            constants.append(DeltaInPad)
            #Hidden params
            if 'J' in inactive:
                J=rescale*torch.ones((self.hiddenSize,self.hiddenSize)).detach_()
                constants.append(J)
            else:
                J=normal((self.hiddenSize,self.hiddenSize)).requires_grad_(True)
                params.append(J)
            #Output params
            if 'WeightOutput' in inactive:
                WOutParam=rescale*torch.ones((self.outputUnits,outputSize)).detach_()
                constants.append(WOutParam)
            else:
                WOutParam=normal((self.outputUnits,self.outputSize)).requires_grad_(True)
                params.append(WOutParam)
            WOutZeroPad=torch.zeros(self.hiddenSize-self.outputUnits,self.outputSize).detach_()
            constants.append(WOutZeroPad)
            if 'DeltaOutput' in inactive:
                DeltaOutParam=torch.zeros(self.outputSize).detach_()
                constants.append(DeltaOutParam)
            else:
                DeltaOutParam=torch.zeros(self.outputSize).requires_grad_(True)
                params.append(DeltaOutParam)
            return (params,constants)
        return get_params

    def get_forward_fn_fun(self,activation:Callable=torch.tanh,isTypical:bool=True):
        '''
        name: get_forward_fn_fun 
        fuction: get the forward function
        param {activation}: the activation function
        return: the function
        '''        
        self.activation=activation
        self.isTypical=isTypical
        def forward_fn(Xs:torch.Tensor,state:tuple,weights:tuple):
            '''
            name: forward_fn
            function: forward function
            param {Xs}: the input
            param {state}: the state
            param {weights}: the weights
            return: the output, the new state
            '''
            params,constants=weights
            if isinstance(params,Tuple):
                params=list(params)
                constants=list(constants)
            params,constants=params.copy(),constants.copy()
            #Input params
            if 'WeightInput' in self.inactive:
                WInParam=constants.pop(0)
            else:
                WInParam=params.pop(0)
            WInZeroPad=constants.pop(0)
            if 'DeltaInput' in self.inactive:
                DeltaInParam=constants.pop(0)
            else:
                DeltaInParam=params.pop(0)
            DeltaInPad=constants.pop(0)
            #Hidden params
            if 'J' in self.inactive:
                J=constants.pop(0)
            else:
                J=params.pop(0)
            #Output params
            if 'WeightOutput' in self.inactive:
                WOutParam=constants.pop(0)
            else:
                WOutParam=params.pop(0)
            WOutZeroPad=constants.pop(0)
            if 'DeltaOutput' in self.inactive:
                DeltaOutParam=constants.pop(0)
            else:
                DeltaOutParam=params.pop(0)
            S,=state
            Ys=[]
            for X in Xs:
                #Calculate H
                H=-torch.mm(X,torch.cat((WInParam,WInZeroPad),dim=1))\
                -torch.cat((DeltaInParam,DeltaInPad),dim=0)+torch.mm(S,J)
                #Calculate S
                if self.isTypical:
                    S=self.activation(H)
                else:
                    S=self.activation(H*S)
                #Calculate Y
                Y=-torch.mm(S,torch.cat((WOutZeroPad,WOutParam),dim=0))+DeltaOutParam
                Ys.append(Y)
            return torch.cat(Ys,dim=0),(S,)
        return forward_fn
    
    def get_predict_fun(self,outputTransoform:Callable=lambda x:x,\
                        interval:int=1):
        '''
        name: get_predict_fun
        fuction: get the function for prediction
        param{outputTransoform}: the transform function for output
        param{interval}: the interval of prediction
        return: the function
        '''        
        self.outputTransoform=outputTransoform
        @torch.no_grad()
        def predict_fun(prefix:torch.Tensor,net:StandardSNN,numPreds:int=1):
            '''
            name: predict_fun
            function: predict the next numPreds
            param {prefix}: the prefix
            param {numPreds}: the number of prediction
            param {net}: the network
            return: the prediction
            '''
            state=net.begin_state(batch_size=1)
            outputs=[pre for pre in prefix[0:interval]]
            get_input=lambda: torch.unsqueeze(outputs[-interval],dim=0)
            #warm-up
            for Y in prefix[interval:]:
                _,state=net(get_input(),state)
                outputs.append(Y)
            for _ in range(numPreds):
                Y,state=net(get_input(),state)
                outputs.append(Y)
            return self.outputTransoform(outputs)
        return predict_fun
    
    @staticmethod
    def grad_clipping(net, theta):
        '''
        name: grad_clipping
        function: clip the gradients
        param {net}: the network
        param {theta}: the threshold
        '''
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = [p for p in net.params if p.requires_grad]
        for param in params:
            param.grad.data.clamp_(-theta, theta)

    @staticmethod
    def train_epoch(net,trainIter,loss,updater,isRandomIter,\
        clipTheta:float=1.0):
        '''
        name: train_epoch
        function: train the network for one epoch
        param {net}: the network
        param {trainIter}: the train iterator
        param {loss}: the loss function
        param {updater}: the updater
        param {isRandomIter}: whether the iterator is random
        param {clipTheta}: the threshold
        '''
        state,timer=None, hp.Timer()
        #sum of training loss, y number
        metric=hp.Accumulator(2)
        for X,Y in trainIter:
            if state is None or isRandomIter:
                state=net.begin_state(batch_size=X.shape[0])
            else:
                if isinstance(net,nn.Module) and not isinstance(state,tuple):
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
            y=Y.transpose(0,1).reshape(-1,Y.shape[-1])
            y_hat,state=net(X,state)
            assert y_hat.shape==y.shape, 'y_hat.shape={}, y.shape={}'.format(y_hat.shape,y.shape)
            l=loss(y_hat,y).mean()
            if isinstance(updater,torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                SuportFunction.grad_clipping(net,clipTheta)
                updater.step()
            else:
                l.backward()
                SuportFunction.grad_clipping(net,clipTheta)
                updater(batch_size=1)
            metric.add(l*y.numel(),y.numel())
        return metric[0]/metric[1], metric[1]/timer.stop()

    @staticmethod
    def evaluate_accuracy(net,testIter,loss,isRandomIter):
        '''
        name: evaluate_accuracy
        function: evaluate the accuracy
        param {net}: the network
        param {testIter}: the test iterator
        param {loss}: the loss function 
        param {isRandomIter}: whether the iterator is random
        '''
        #sum of testing loss, y number
        state=None
        metric=hp.Accumulator(2)
        for X,Y in testIter:
            if state is None or isRandomIter:
                state=net.begin_state(batch_size=X.shape[0])
            else:
                if isinstance(net,nn.Module) and not isinstance(state,tuple):
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
            y=Y.transpose(0,1).reshape(-1,Y.shape[-1])
            y_hat,state=net(X,state)
            assert y_hat.shape==y.shape, 'y_hat.shape={}, y.shape={}'.format(y_hat.shape,y.shape)
            l=loss(y_hat,y).mean()
            metric.add(l*y.numel(),y.numel())
        return metric[0]/metric[1]
            
