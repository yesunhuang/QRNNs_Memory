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
from sqlalchemy import case
import torch
from torch import nn, pi
from collections.abc import Callable
import numpy as np
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
        return self.forward_fn(X.transpose(0,1),state,(self.params,self.constants))
    
    def call_with_weight(self, X: torch.Tensor, weight: tuple):
        '''
        name:call_with_weight
        function: call the quantum SRNN with weight
        param {X}: the input
        param {weight}: the weight
        return: the output
        '''
        Y,_=self.forward_fn(X.transpose(0,1),\
            self.begin_state(X.shape[0]),(weight,self.constants))
        Y=Y.reshape((-1,X.shape[0],Y.shape[-1])).transpose(0,1)
        return Y

    def begin_state(self,batch_size:int=1):
        '''
        name:begin_state
        function: begin the state
        param {batch_size}: the batch size
        return: the state
        '''
        return self.init_state(batch_size,self.qubits)

class QuantumSystemFunction:
    '''The functions depicting the used quantum system'''

    def __init__(self):
        '''
        name:__init__
        function: initialize the class for quantum system function
        '''
    
    def get_init_state_fun(self,activation:list=[],isDensity:bool=False):
        '''
        name: 
        fuction: 
        param {activation}: a list describe the pattern of activation
        param {isDensity}: whether the state is density matrix  
        return: the init_state function
        '''        
        self.activation=activation

        def init_state(batch_size:int, qubits:int):
            '''
            name:init_state
            function: initialize the state
            param {batch_size}: the batch size
            param {qubits}: the number of qubits
            return: the state
            '''
            state=[]
            for i in range(0,qubits):
                if i in self.activation:
                    if isDensity:
                        state.append(qt.fock_dm(2,1))
                    else:
                        state.append(qt.basis(2,1))
                else:
                    if isDensity:
                        state.append(qt.fock_dm(2,0))
                    else:
                        state.append(qt.basis(2,0))
            state=qt.tensor(state)
            S=[state.copy() for _ in range(batch_size)]
            return (S,)
        return init_state

    def get_get_params_fun(self, inputQubits:list=[],outputQubits:list=[],\
                        interQPairs:list=[],inactive:list=[],\
                        rescale:dict={}):
        '''
        name:get_get_params_fun
        function: get the get_params function
        param {inputQubits}: the input qubits
        param {outputQubits}: the output qubits
        param {interQPairs}: the interaction qubit pairs
        param {rescale}: the rescale the initial parameters
        param {inactive}: the inactive params
        'WIn','DeltaIn','J',''WOut','DeltaOut'
        return: the get_params function
        '''
        self.inputQubits,self.outputQubits=inputQubits,outputQubits
        self.interQPairs,self.rescale,self.inactive=interQPairs,rescale,inactive
        defaultRescale={'WIn':0.01,'DeltaIn':0,'J':torch.tensor([0.01]),'WOut':0.01,'DeltaOut':0}
        for key in rescale.keys():
            if key not in defaultRescale.keys():
                raise ValueError('The rescale key is not in the default rescale')
            defaultRescale[key]=rescale[key]
        self.rescale=defaultRescale

        def normal(shape,scale:float=1.0):
            '''
            name: normal
            function: get the normal distribution
            param {shape}: the shape of the distribution
            param {rescale}: the rescale of the distribution
            return: the distribution
            '''
            return torch.randn(size=shape)*scale

        def get_params(inputSize:int,qubits:int,outputSize:int):
            '''
            name:get_params
            function: get the parameters
            param {inputSize}: the size of input
            param {qubits}: the number of qubits
            param {outputSize}: the size of output
            return: the parameters
            '''
            params=[]
            constants=[]
            assert len(inputQubits)<=qubits, 'Too many input qubits'
            assert len(outputQubits)<=qubits, 'Too many output qubits'
            self.inputSize,self.qubits,self.outputSize=inputSize,qubits,outputSize
            #Input params
            if 'WIn' in self.inactive:
                WIn=rescale['WIn']*torch.ones((inputSize,len(self.inputQubits))).detach_()
                constants.append(WIn)
            else:
                WIn=normal((inputSize,len(self.inputQubits)),rescale['WIn']).requires_grad_(True)
                params.append(WIn)
            if 'DeltaIn' in self.inactive:
                DeltaInParam=rescale['DeltaIn']*torch.ones(len(self.inputQubits)).detach_()
                constants.append(DeltaInParam)
            else:
                DeltaInParam=rescale['DeltaIn']*torch.ones(len(self.inputQubits)).requires_grad_(True)
                params.append(DeltaInParam)
            DeltaInPad=rescale['DeltaIn']*torch.ones(qubits-len(self.inputQubits)).detach_()
            constants.append(DeltaInPad)
            #Interaction params
            if 'J' in self.inactive:
                J=rescale['J']*torch.ones((len(self.interQPairs),)).detach_()
                constants.append(J)
            else:
                J=normal((len(self.interQPairs),),rescale['J']).requires_grad_(True)
                params.append(J)
            #Output params
            if 'WOut' in self.inactive:
                WOut=rescale['WOut']*torch.ones((len(self.outputQubits),outputSize)).detach_()
                constants.append(WOut)
            else:
                WOut=normal((len(self.outputQubits),outputSize),rescale['WOut']).requires_grad_(True)
                params.append(WOut)
            if 'DeltaOut' in self.inactive:
                DeltaOutParam=rescale['DeltaOut']*torch.ones((len(self.outputQubits),outputSize)).detach_()
                constants.append(DeltaOutParam)
            else:
                DeltaOutParam=normal((len(self.outputQubits),outputSize),rescale['DeltaOut']).requires_grad_(True)
                params.append(DeltaOutParam)
            return (params,constants)
        return get_params    

    def get_forward_fn_fun(self,sysConstants:dict,samples:int=None,measEffect:bool=False):
        '''
        name:get_forward_fn_fun
        function: get the forward function
        param {sysConstants}: the system constants
        param {samples}: the number of samples
        param {measEffect}: whether the measurement effect is included
        return: the forward function
        '''
        self.samples,self.measEffect=samples,measEffect
        self.sysConstants=sysConstants

        def multi_qubits_sigma(pauliBasis:list=['i']):
            '''
            name:build_multi_qubits_sigma
            function: build the multi qubits sigma
            param {pauliBasis}: the pauliBasis
            return: the sigma
            '''
            sigma=[]
            for i in range(0,len(pauliBasis)):
                if pauliBasis[i]=='i':
                    sigma.append(qt.qeye(2))
                if pauliBasis[i]=='x':
                    sigma.append(qt.sigmax())
                if pauliBasis[i]=='y':
                    sigma.append(qt.sigmay())
                if pauliBasis[i]=='z':
                    sigma.append(qt.sigmaz())
                if pauliBasis[i]=='-':
                    sigma.append(qt.sigmam())
                if pauliBasis[i]=='+':
                    sigma.append(qt.sigmap())
            return qt.tensor(sigma)

        def build_int_operators(J:torch.Tensor,qubits:int):
            '''
            name:build_int_operators
            function: build the interaction operators
            param {J}: the interaction strength
            return: the interaction operators
            '''
            H_I=[]
            C_ops=[]
            assert len(J)==self.interQPairs, \
                'The length of J is not equal to the number of interaction pairs'
            for i in range(0,qubits):
                pauliBasis=['i']*qubits
                if 'Dissipation' in self.sysConstants:
                    pauliBasis[i]='-'
                    sigma=multi_qubits_sigma(pauliBasis)
                    C_ops.append(np.sqrt(self.sysConstants['Dissipation'])*sigma)
                pauliBasis[i]='x'
                sigma=multi_qubits_sigma(pauliBasis)
                H_I.append([self.sysConstants['Omega']/2.0,sigma])
           
            for i in range(0,len(self.interQPairs)):
                pauliBasis=['i']*qubits
                pauliBasis[self.interQPairs[i][0]]='x'
                pauliBasis[self.interQPairs[i][1]]='x'
                sigma=multi_qubits_sigma(pauliBasis)
                H_I.append([J[i],sigma])
            return H_I,C_ops

        def encode_input(x:torch.Tensor,inputParams:tuple):
            '''
            name: encodeInput
            function: encode the input
            param {x}: the input
            param {inputParams}: the input parameters
            return: the encoded input
            '''
            #TODO: encode the input into Hamiltonian and dissipation
            pass

        def evolve(S:list,evolParam:tuple):
            '''
            name: evolve
            function: evolve the state
            param {S}: the state
            param {evolParam}: the evolution parameters
            return: the evolved state
            '''
            #TODO: evolve the state
            pass

        def measure(S:list):
            '''
            name: measure
            function: measure the state
            param {S}: the state
            return: the measured state
            '''
            #TODO: measure the state
            pass
        
        def forward_fn(Xs:torch.tensor,state:tuple,weights:tuple):
            '''
            name:forward_fn
            function: the forward function
            param {Xs}: the input
            param {state}: the state
            param {weights}: the weights
            return: the output
            '''
            params,constants=weights
            if isinstance(params,tuple):
                params=list(params)
                constants=list(constants)
            params,constants=params.copy(),constants.copy()
            #Input params
            if 'WIn' in self.inactive:
                WIn=constants.pop(0)
            else:
                WIn=params.pop(0)
            if 'DeltaIn' in self.inactive:
                DeltaInParam=constants.pop(0)
            else:
                DeltaInParam=params.pop(0)
            DeltaInPad=constants.pop(0)
            qubits=DeltaInPad.shape[0]+DeltaInParam[0]
            inputParams=(WIn,DeltaInParam,DeltaInPad)
            #Interaction params
            if 'J' in self.inactive:
                J=constants.pop(0)
            else:
                J=params.pop(0)
            #Output params
            if 'WOut' in self.inactive:
                WOut=constants.pop(0)
            else:
                WOut=params.pop(0)
            if 'DeltaOut' in self.inactive:
                DeltaOutParam=constants.pop(0)
            else:
                DeltaOutParam=params.pop(0)
            S,=state
            Ys=[]
            H_I,Co_ps=build_int_operators(J,qubits)
            for X in Xs:
                H_input=encode_input(X,inputParams)
                H=H_input+H_I
                S=evolve(S,H,Co_ps)
                S,measResult=measure(S)
                Y=torch.mm(WOut,measResult)+DeltaOutParam
                Ys.append(Y)
            return torch.cat(Ys,dim=0),(S,)
        return forward_fn



        
            