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
from torch import pi
from collections.abc import Callable
import numpy as np
import qutip as qt
import qutip.measurement as qtm
#modify path
from GradientFreeOptimizers.CostFunc import StandardSNN

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
            if self.samples==None:
                S=[state.copy() for _ in range(batch_size)]
            else:
                S=[[state.copy() for _ in range(self.samples)] for _ in range(self.samples)]
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
            {'WIn':0.01,'DeltaIn':0,'J':torch.tensor([0.01]),'WOut':0.01,'DeltaOut':0}
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
            DeltaInPad=[qubits,rescale['DeltaIn']]
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

    def get_forward_fn_fun(self,sysConstants:dict={},samples:int=None,measEffect:bool=False):
        '''
        name:get_forward_fn_fun
        function: get the forward function
        param {sysConstants}: the system constants
            {'measureQuantity':'z',
                            'Dissipation':'0.0',
                            'Omega':1.0,
                            'tau':1.51*pi,
                            'steps':10,
                            'options':qt.Options()}
        param {samples}: the number of samples
        param {measEffect}: whether the measurement effect is included
        return: the forward function
        '''
        self.samples,self.measEffect=samples,measEffect
        self.measOperators=build_measure_operators()
        defaultSysConstants={'measureQuantity':'z',\
                            'Dissipation':'0.0',\
                            'Omega':1.0,\
                            'tau':1.51*pi,\
                            'steps':10,\
                            'numCpus':1,\
                            'options':qt.Options()}
        for key in sysConstants.keys():
            if key not in defaultSysConstants.keys():
                raise ValueError('The sysConstants key is not in the default sysConstants')
            defaultSysConstants[key]=sysConstants[key]
        self.sysConstants=defaultSysConstants

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
        
        def build_measure_operators(qubits:int):
            '''
            name:build_measure_operators
            function: build the measure operators
            return: the measure operators
            '''
            measureOperators=[]
            M=self.sysConstants['measureQuantity']
            pauliBasis=['i']*qubits
            for i in range(0,self.outputQubits):
                pauliBasis[i]=M
                measureOperators.append(multi_qubits_sigma(pauliBasis))
            return measureOperators

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

        def encode_input(xBatch:torch.Tensor,inputParams:tuple,qubits:int):
            '''
            name: encodeInput
            function: encode the input
            param {xBatch}: the input
            param {inputParams}: the input parameters
            return: the encoded input
            '''
            H_input=[]
            WIn,DeltaIn,DeltaInPad=inputParams
            DeltaEncoded=torch.mm(xBatch,WIn)+DeltaIn
            sigmazList=[]
            for i in range(0,qubits):
                pauliBasis=['i']*qubits
                pauliBasis[i]='z'
                sigmazList.append(multi_qubits_sigma(pauliBasis))
            for delta in DeltaEncoded:
                H=[]
                for i in range(0,qubits):
                    if i in self.inputQubits:
                        H.append([delta[i].item(),sigmazList[i].copy()])
                    else:
                        H.append([DeltaInPad[1],sigmazList[i].copy()])
                H_input.append(H.copy())
            return H_input

        def evolve(S:tuple,H:tuple,Co_ps:list):
            '''
            name: evolve
            function: evolve the state
            param {S}: the state
            param {evolParam}: the evolution parameters
            return: the evolved state
            '''
            def density_mesolve(value:tuple):
                '''
                name: densityMesolve
                function: evolve the system via mesolve
                param {S}:the initial state
                return: the evolved state
                '''
                Hs,rho0=value
                tlist=np.linspace(0,self.sysConstants['tau'],self.sysConstants['steps'])
                finalState=qt.mesolve(rho0,Hs,Co_ps,tlist,options=self.sysConstants['options']).states[-1]
                return finalState
            
            def state_mcsolve(value:tuple):
                '''
                name: stateMcsolve
                function: evolve the system via mcsolve
                param {value}:(single Hamiltonian,the initial state)
                return: the evolved state
                '''
                def single_mc(sampleState):
                    return qt.mcsolve(sampleState,Hs,Co_ps,tlist,ntraj=1,\
                    options=self.sysConstants['options'],progress_bar=False).states[-1]
                Hs,rho0s=value
                tlist=np.linspace(0,self.sysConstants['tau'],self.sysConstants['steps'])
                finalState=[single_mc(rho0) for rho0 in rho0s]
                #finalState=qt.parallel_map(single_mc,rho0s)
                return finalState

            state,=S
            H_I,H_input=H
            values=[(singleH+H_I,singleState) for singleH,singleState in zip(H_input,state)]
            if self.samples==None:
                if self.sysConstants['numCpus']==1:
                    result=[density_mesolve(value) for value in values]
                else:
                    result=qt.parallel_map(density_mesolve,values,num_cpus=self.sysConstants['numCpus'])
            else:
                if self.sysConstants['numCpus']==1:
                    result=[state_mcsolve(value) for value in values]
                else:
                    result=qt.parallel_map(state_mcsolve,values,num_cpus=self.sysConstants['numCpus'])
            return (result,)

        def measure(S:tuple,qubits:int):
            '''
            name: measure
            function: measure the state
            param {S}: the state
            return: (the measured state,the measured result)
            '''

            def density_measure(state:qt.Qobj):
                '''
                name: densityMeasure
                function: measure the system via mesolve
                param {value}:the initial state
                return: the measured state,the measured result
                '''
                measResult=[]
                for measOp in self.measOperators:
                    measResult.append(qt.expect(measOp,state))
                if not self.measEffect:
                    return state,measResult
                else:
                    newState=None    
                    for measOp in self.measOperators:
                        _,projectors,probabilities=qtm.measure_observable(state,measOp)
                        for proj,prob in zip(projectors,probabilities):
                            if newState==None:
                                newState=proj*state*proj.dag()*prob
                            else:
                                newState=newState+proj*prob
                    return [newState,measResult]
            def state_measure(state:list):
                '''
                name: stateMeasure
                function: measure the system via mcsolve
                param {value}:the initial state
                return: the measured state,the measured result
                '''
                measResult=[0.0]*len(self.measOperators)
                for i in range(len(self.measOperators)):
                    for j in range(len(state)):
                        if self.measEffect:
                            measValue,state[j]=qtm.measure_state(state[j],self.measOperators[i])
                        else:
                            measValue,_=qtm.measure_state(state[j],self.measOperators[i])
                        measResult[i]=measResult[i]+measValue
                return state,[value/len(state) for value in measResult]

            stateBatch,=S
            if self.samples==None:
                if self.sysConstants['numCpus']==1:
                    result=[density_measure(singleState) for singleState in stateBatch]
                else:
                    result=qt.parallel_map(density_measure,stateBatch,num_cpus=self.sysConstants['numCpus'])
                measState=(result[:,0],)
                measResult=result[:,1]
                return measState,torch.tensor(measResult)
            else:
                if self.sysConstants['numCpus']==1:
                    result=[state_measure(singleState) for singleState in stateBatch]
                else:
                    result=qt.parallel_map(state_measure,stateBatch,num_cpus=self.sysConstants['numCpus'])
                measState=(result[:,0],)
                measResult=result[:,1]
                return measState,torch.tensor(measResult)

        
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
            qubits=DeltaInPad[0]
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
                H_input=encode_input(X,inputParams,qubits)
                S=evolve(S,(H_I,H_input),Co_ps)
                S,measResult=measure(S)
                Y=torch.mm(WOut,measResult)+DeltaOutParam
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
          
