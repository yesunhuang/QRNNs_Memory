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
import torch
from torch import nn
#modify path
from GradientFreeOptimizers.CostFunc import StandardSNN
import GradientFreeOptimizers.Helpers as hp

class ClassicalSRNN(StandardSNN):
    '''implement the classical sRNN'''

    def __init__(self, inputSize:int=1, hiddenSize:int=10, outputSize:int=1,\
                get_params:function=None,init_state:function=None, forward_fn:function=None):
        '''
        name: __init__ 
        fuction: initialize the class for classical SRNN
        param {inpoutSize}: the size of input
        param {hiddenSize}: the size of hidden units
        param {outputSize}: the size of output
        param {getParams}: the function to get parameters
        param {initState}: the function to initialize the state
        param {forwardFn}: the function to forward
        '''
        self.inputSize,self.hiddenSize,self.outputSize = inputSize,hiddenSize,outputSize
        (self.params,self.constants)=get_params(inputSize,hiddenSize,outputSize)
        self.init_state,self.forward_fn = init_state,forward_fn
    
    def __call__(self,X,state):
        '''
        name: __call__
        function: call the class
        param {input}: the input
        param {state}: the state
        return: the output, the new state
        '''
        return self.forward_fn(X.T,state,(self.params,self.constants))

    def call_with_weight(self, X: torch.Tensor, weight: tuple):
        '''
        name: call_with_weight
        function: call the class with weight
        param {X}: the input
        param {weight}: the params and state (param,state)
        return: the output
        '''
        return self.forward_fn(X.T,weight[1],weight[0])
    
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
            return torch.full((batch_size,hiddenSize),self.initStateValue)
        return init_state

    def get_get_params_fun(self,inputRatio:float=1.0, outputRatio:float=1.0,\
                            rescale:float=0.01):
        '''
        name: get_get_params_fun
        function: get the function to get the parameters
        param {inputRatio}: the ratio of input units
        param {outputRatio}: the ratio of output units
        return: the function
        '''
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
            #Input params
            WInParam=normal((inputSize,self.inputUnits)).requires_grad_(True)
            WInZeroPad=torch.zeros((inputSize,self.hiddenSize-self.inputUnits)).detach_()
            DeltaInParam=torch.zeros(self.inputUnits).requires_grad_(True)
            DeltaInPad=torch.zeros(self.hiddenSize-self.inputUnits).detach_()
            #Hidden params
            J=normal((self.hiddenSize,self.hiddenSize)).requires_grad_(True)
            #Output params
            WOutParam=normal(self.outputUnits,self.outputSize).requires_grad_(True)
            WOutZeroPad=torch.zeros(self.hiddenSize-self.outputUnits,self.outputSize).detach_()
            DeltaOutParam=torch.zeros(self.outputUnits).requires_grad_(True)
            DeltaOutPad=torch.zeros(self.hiddenSize-self.outputUnits).detach_()
            #Group params
            params=(WInParam,DeltaInParam,J,WOutParam,DeltaOutParam)
            constants=(WInZeroPad,DeltaInPad,WOutZeroPad,DeltaOutPad)
            return (params,constants)
        return get_params

    def get_forward_fn_fun(self,activation:function=torch.tanh):
        '''
        name: get_forward_fn_fun 
        fuction: get the forward function
        param {activation}: the activation function
        return: the function
        '''        
        self.activation=activation
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
            WInParam,DeltaInParam,J,WOutParam,DeltaOutParam=params
            WInZeroPad,DeltaInPad,WOutZeroPad,DeltaOutPad=constants
            S,=state
            Ys=[]
            for X in Xs:
                #Calculate H
                H=-torch.mm(X,torch.cat((WInParam,WInZeroPad),dim=1))\
                -torch.cat((DeltaInParam,DeltaInPad),dim=0)+torch.mm(S,J)
                #Calculate S
                S=self.activation(H)
                #Calculate Y
                Y=-torch.mm(S,torch.cat((WOutZeroPad,WOutParam),dim=1))\
                +torch.cat((DeltaOutPad,DeltaOutParam),dim=0)
                Ys.append(Y)
            return torch.cat(Ys,dim=0),(S,)
        return forward_fn
    
    def get_predict_fun(self,outputTransoform:function=lambda x:x):
        '''
        name: get_predict_fun
        fuction: get the function for prediction
        return: the function
        '''        
        self.outputTransoform=outputTransoform
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
            outputs=[prefix[0]]
            #warm-up
            for Y in prefix[1:]:
                _,state=net(Y,state)
                outputs.append(Y)
            for _ in range(numPreds):
                Y,state=net(outputs[-1],state)
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
            params=net.params
        for param in params:
            param.grad.data.clamp_(-theta, theta)

    @staticmethod
    def train_epoch(net,trainIter,loss,updater,isRandomIter,\
        clipTheta:float=1.0,):
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
            y=Y.T.reshape(-1)
            y_hat,state=net(X,state)
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
            
