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
    
    def __init_state(self,batch_size:int, qubits:int):
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
                if self.isDensity:
                    state.append(qt.fock_dm(2,1))
                else:
                    state.append(qt.basis(2,1))
            else:
                if self.isDensity:
                    state.append(qt.fock_dm(2,0))
                else:
                    state.append(qt.basis(2,0))
        state=qt.tensor(state)
        #print(state)
        if self.isDensity==True:
            S=[state.copy() for _ in range(batch_size)]
        else:
            S=[[state.copy() for _ in range(self.samples)] for _ in range(batch_size)]
        return (S,)

    def get_init_state_fun(self,activation:list=[],isDensity:bool=True):
        '''
        name: 
        fuction: 
        param {activation}: a list describe the pattern of activation
        param {isDensity}: whether the state is density matrix  
        return: the init_state function
        '''        
        self.activation=activation
        self.isDensity=isDensity
        return self.__init_state

    def normal(self,shape,scale:float=1.0):
        '''
        name: normal
        function: get the normal distribution
        param {shape}: the shape of the distribution
        param {scale}: the rescale of the distribution
        return: the distribution
        '''
        return torch.randn(size=shape)*scale

    def ones(self,shape,scale:float=1.0):
        '''
        name: scalar
        function: get the scalar distribution
        param {shape}: the shape of the distribution
         param {scale}: the rescale of the distribution
        return: the ones
        '''
        return torch.ones(shape)*scale

    def __get_params(self,inputSize:int,qubits:int,outputSize:int):
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
        assert len(self.inputQubits)<=qubits, 'Too many input qubits'
        assert len(self.outputQubits)<=qubits, 'Too many output qubits'
        self.inputSize,self.qubits,self.outputSize=inputSize,qubits,outputSize
        if self.isRandom:
            init_value=self.normal
        else:
            init_value=self.ones
        #Input params
        if 'WIn' in self.inactive:
            WIn=init_value((inputSize,len(self.inputQubits)),self.rescale['WIn']).detach_()
            constants.append(WIn)
        else:
            WIn=init_value((inputSize,len(self.inputQubits)),self.rescale['WIn']).requires_grad_(True)
            params.append(WIn)
        if 'DeltaIn' in self.inactive:
            DeltaInParam=self.ones(len(self.inputQubits),self.rescale['DeltaIn']).detach_()
            constants.append(DeltaInParam)
        else:
            DeltaInParam=self.ones(len(self.inputQubits),self.rescale['DeltaIn']).requires_grad_(True)
            params.append(DeltaInParam)
        DeltaInPad=[qubits,self.rescale['DeltaIn']]
        constants.append(DeltaInPad)
        #Interaction params
        if 'J' in self.inactive:
            J=init_value((len(self.interQPairs),1),self.rescale['J']).detach_()
            constants.append(J)
        else:
            J=init_value((len(self.interQPairs),1),self.rescale['J']).requires_grad_(True)
            params.append(J)
        #Output params
        if 'WOut' in self.inactive:
            WOut=init_value((len(self.outputQubits),outputSize),self.rescale['WOut']).detach_()
            constants.append(WOut)
        else:
            WOut=init_value((len(self.outputQubits),outputSize),self.rescale['WOut']).requires_grad_(True)
            params.append(WOut)
        if 'DeltaOut' in self.inactive:
            DeltaOutParam=self.ones(outputSize,self.rescale['DeltaOut']).detach_()
            constants.append(DeltaOutParam)
        else:
            DeltaOutParam=self.ones(outputSize,self.rescale['DeltaOut']).requires_grad_(True)
            params.append(DeltaOutParam)
        return (params,constants)

    def get_get_params_fun(self, inputQubits:list=[],outputQubits:list=[],\
                        interQPairs:list=[],inactive:list=[],\
                        rescale:dict={},isRandom:bool=True):
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
        param {isRandom}: whether the parameters are random initialized
        return: the get_params function
        '''
        self.inputQubits,self.outputQubits=inputQubits,outputQubits
        self.interQPairs,self.inactive=interQPairs,inactive
        self.isRandom=isRandom
        defaultRescale={'WIn':0.01,'DeltaIn':0,'J':torch.tensor([0.01]),'WOut':0.01,'DeltaOut':0}
        for key in rescale.keys():
            if key not in defaultRescale.keys():
                raise ValueError('The rescale key is not in the default rescale')
            defaultRescale[key]=rescale[key]
        self.rescale=defaultRescale
        return self.__get_params    

    def __density_mesolve(self,value:tuple):
        '''
        name: densityMesolve
        function: evolve the system via mesolve
        param {value}:(single Hamiltonian,the initial density matrices)
        return: the evolved state
        '''
        Hs,rho0,Co_ps=value
        #print(len(Co_ps))
        #print(type(Hs))
        #print(type(rho0))
        tlist=np.linspace(0,float(self.sysConstants['tau']),int(self.sysConstants['steps']))
        finalState=qt.mesolve(Hs,rho0,tlist,c_ops=Co_ps,options=self.sysConstants['options']).states[-1]
        return finalState

    def __single_mc(self,sampleState,Co_ps,Hs,tlist):
            if len(Co_ps)==0:
                state=qt.sesolve(Hs,sampleState,tlist,\
                    options=self.sysConstants['options'],progress_bar=None).states[-1]
            else:
                state=qt.mcsolve(Hs,sampleState,tlist,c_ops=Co_ps,ntraj=1,\
                    options=self.sysConstants['options'],progress_bar=None).states[0,-1]
            assert isinstance(state,qt.Qobj), 'The evolved state is not a quantum object'
            return state
    
    def __state_mcsolve(self,value:tuple):
        '''
        name: stateMcsolve
        function: evolve the system via mcsolve
        param {value}:(single Hamiltonian,the initial states)
        return: the evolved state
        ''' 
        Hs,rho0s,Co_ps=value
        tlist=np.linspace(0,float(self.sysConstants['tau']),int(self.sysConstants['steps']))
        finalState=[self.__single_mc(rho0,Co_ps,Hs,tlist) for rho0 in rho0s]
        #finalState=qt.parallel_map(single_mc,rho0s)
        return finalState

    def __density_measure(self,state:qt.Qobj):
        '''
        name: densityMeasure
        function: measure the system via mesolve
        param {value}:the initial state
        return: the measured state,the measured result
        '''
        measResults=[]
        newState=state.copy()
        for measValues,measOp in zip(self.measOperators[0],self.measOperators[1]):
            #print(measValues,measOp)
            collapsedStates,probabilities=qtm.measurement_statistics_povm(newState,measOp)
            measResult=0.0
            for measValue,probability in zip(measValues,probabilities):
                measResult+=measValue*np.real(probability) 
            measResults.append(measResult)
            if self.measEffect:
                newState=0
                for collapsedState,probability in zip(collapsedStates,probabilities):
                    if not collapsedState==None:
                        newState+=collapsedState*probability
        return newState,measResults

    def __state_measure(self,state:list):
        '''
        name: stateMeasure
        function: measure the system via mcsolve
        param {value}:the initial state
        return: the measured state,the measured result
        '''
        measResults=[0.0]*len(self.measOperators[1])
        for i in range(len(self.measOperators)):
            for j in range(len(state)):
                #print(state[j])
                if self.measEffect:
                    measIndex,state[j]=qtm.measure_povm(state[j],self.measOperators[1][i])
                    #print(j)
                else:
                    measIndex,_=qtm.measure_povm(state[j],self.measOperators[1][i])
                measResults[i]=measResults[i]+self.measOperators[0][i][int(measIndex)]
        return state,[value/len(state) for value in measResults]

    def __multi_qubits_sigma(self,pauliBasis:list=['i']):
        '''
        name:build_multi_qubits_sigma
        function: build the multi qubits sigma
        param {pauliBasis}: the pauliBasis
        return: the sigma
        '''
        sigma=[]
        for i in range(0,len(pauliBasis)):
            if pauliBasis[i]=='i':
                sigma.append(qt.identity(2))
            elif pauliBasis[i]=='x':
                sigma.append(qt.sigmax())
            elif pauliBasis[i]=='y':
                sigma.append(qt.sigmay())
            elif pauliBasis[i]=='z':
                sigma.append(qt.sigmaz())
            elif pauliBasis[i]=='-':
                sigma.append(qt.sigmam())
            elif pauliBasis[i]=='+':
                sigma.append(qt.sigmap())
        return qt.tensor(sigma)
    
    def __build_measure_operators(self,qubits:int):
        '''
        name:build_measure_operators
        function: build the measure operators
        return: the measure operators
        '''
        measureOperators=[]
        M=self.sysConstants['measureQuantity']
        if M=='z':
            sigma=qt.sigmaz()
        elif M=='x':
            sigma=qt.sigmax()
        elif M=='y':
            sigma=qt.sigmay()
        eigenvalues,projectors,_=qtm.measurement_statistics_observable(qt.fock_dm(2,0),sigma)
        measureValues=[eigenvalues]*len(self.outputQubits)
        for i in range(0,len(self.outputQubits)):
            P1=[]
            P2=[]
            for j in range(qubits):
                if j==self.outputQubits[i]:
                    P1.append(projectors[0])
                    P2.append(projectors[1])
                else:
                    P1.append(qt.identity(2))
                    P2.append(qt.identity(2))
            P1=qt.tensor(P1)
            P2=qt.tensor(P2)
            measureOperators.append([P1,P2])
        return [measureValues,measureOperators]

    def __build_int_operators(self,J:torch.Tensor,qubits:int):
        '''
        name:build_int_operators
        function: build the interaction operators
        param {J}: the interaction strength
        return: the interaction operators
        '''
        H_I=0
        C_ops=[]
        assert J.numel()==len(self.interQPairs), \
            'The length of J is not equal to the number of interaction pairs'
        for i in range(0,qubits):
            pauliBasis=['i']*qubits
            if 'Dissipation' in self.sysConstants:
                pauliBasis[i]='-'
                sigma=self.__multi_qubits_sigma(pauliBasis)
                C_ops.append(np.sqrt(float(self.sysConstants['Dissipation']))*sigma)
            pauliBasis[i]='x'
            sigma=self.__multi_qubits_sigma(pauliBasis)
            H_I+=float(self.sysConstants['Omega']/2.0)*sigma
           
        for i in range(0,len(self.interQPairs)):
            pauliBasis=['i']*qubits
            pauliBasis[self.interQPairs[i][0]]='x'
            pauliBasis[self.interQPairs[i][1]]='x'
            sigma=self.__multi_qubits_sigma(pauliBasis)
            H_I+=J[i].item()*sigma
        return H_I,C_ops

    def __encode_input(self,xBatch:torch.Tensor,inputParams:tuple,qubits:int):
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
            sigmazList.append(self.__multi_qubits_sigma(pauliBasis))
        for delta in DeltaEncoded:
            H=0
            for i in range(0,qubits):
                if i in self.inputQubits:
                    H+=sigmazList[i]*delta[i].item()
                else:
                    H+=sigmazList[i]*DeltaInPad[1]
            H_input.append(H.copy())
        return H_input

    def __evolve(self,S:list,H:tuple,Co_ps:list):
        '''
        name: evolve
        function: evolve the state
        param {S}: the state
        param {evolParam}: the evolution parameters
        return: the evolved state
        '''
        stateBatch=S
        #print(stateBatch[0])
        H_I,H_input=H
        #print(H_I)
        assert  len(H_input)==len(stateBatch), 'The length of H_input is not equal to the length of stateBatch'
        #print(type(H_input))
        values=[(singleH+H_I,singleState,Co_ps) for singleH,singleState in zip(H_input,stateBatch)]
        #print(type(values[0][1]))
        if self.isDensity==True:
            result=[self.__density_mesolve(value) for value in values]
        else:
            result=[self.__state_mcsolve(value) for value in values]
        return result

    def __measure(self,S:list):
        '''
        name: measure
        function: measure the state
        param {S}: the states
        return: (the measured state,the measured result)
        '''
        stateBatch=S
        if self.isDensity==True:
            results=[self.__density_measure(singleState) for singleState in stateBatch]
        else:
            results=[self.__state_measure(singleState) for singleState in stateBatch]
        measStates=[result[0] for result in results]
        measResults=[result[1] for result in results]
        #print(len(measResults))
        #print(type(measResults[0][0]))
        #print(type(measStates[0]))
        return measStates,torch.tensor(measResults,dtype=torch.float32)
    
    def sub_forward_fn(self,valuePack:tuple):
        '''
        name: sub_forward_fn
        function: for batch parallel
        param {valuePack}:
            param {xSubBatch}: the input
            param {inputParams}: the input parameters
            param {qubits}: the number of qubits
            param {S}: the states
            param {H_I}: the interaction operators
            param {Co_ps}: the collapse operators
        '''
        XSubBatch,inputParams,qubits,S,H_I,Co_ps=valuePack
        measResults=[]
        for X in XSubBatch:
            H_input=self.__encode_input(X,inputParams,qubits)
            S=self.__evolve(S,(H_I,H_input),Co_ps)
            S,measResult=self.__measure(S)
            #Y=torch.mm(measResult,WOut)+DeltaOutParam
            #Ys.append(Y)
            measResults.append(torch.unsqueeze(measResult,0))
        results=torch.cat(measResults,dim=0)
        #assert results[1,0,0]==measResults[1][0,0],'The result is not correct'
        #return results
        return results,S
    def __forward_fn(self,Xs:torch.tensor,state:tuple,weights:tuple):
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
            
        if self.measOperators==None:
            self.measOperators=self.__build_measure_operators(qubits)
        S,=state
        #Ys=[]
        H_I,Co_ps=self.__build_int_operators(J,qubits)
        if self.sysConstants['numCpus']==1:
            value_pack=Xs,inputParams,qubits,S,H_I,Co_ps
            measResults,S=self.sub_forward_fn(value_pack)
        else:
            subBatch=max(1,Xs.shape[1]//self.sysConstants['numCpus'])
            XSubBatchs=list(Xs.split(subBatch,dim=1))
            SSubBatchs=hp.list_split(S,subBatch)
            assert len(XSubBatchs)==len(SSubBatchs),\
                 'The length of XSubBatchs is not equal to the length of SSubBatchs'
            value_packs=[(XSubBatch,inputParams,qubits,SSubBatch,H_I,Co_ps) \
                for (XSubBatch,SSubBatch) in zip(XSubBatchs,SSubBatchs)]
            results=qt.parallel_map(self.sub_forward_fn,value_packs,num_cpus=self.sysConstants['numCpus'])
            measResults=[result[0] for result in results]
            states=[result[1] for result in results]
            S=hp.list_concat(states)
            assert len(S)==Xs.shape[1],'The batch size of states is not correct.'
            measResults=torch.cat(measResults,dim=1)
            assert measResults.shape==torch.Size([Xs.shape[0],Xs.shape[1],len(self.outputQubits)]),\
                'The shape of the measResults is wrong'
        Ys=torch.mm(measResults.reshape(-1,measResults.shape[-1]),WOut)+DeltaOutParam
        #return torch.cat(Ys,dim=0),(S,)
        return Ys,(S,)

    def get_forward_fn_fun(self,sysConstants:dict={},samples:int=1,measEffect:bool=False):
        '''
        name:get_forward_fn_fun
        function: get the forward function
        param {sysConstants}: the system constants
            {'measureQuantity':'z',
            'Dissipation':'0.1',
            'Omega':1.0,
            'tau':1.51*pi,
            'steps':10,
            'numCpus':1,
            'options':qt.Options()}
        param {samples}: the number of samples
        param {measEffect}: whether the measurement effect is included
        return: the forward function
        '''
        self.samples,self.measEffect=samples,measEffect
        self.measOperators=None
        defaultSysConstants={'measureQuantity':'z',\
                            'Dissipation':'0.1',\
                            'Omega':1.0,\
                            'tau':1.51*pi,\
                            'steps':10,\
                            'numCpus':1,\
                            'options':qt.Options()}
        for key in sysConstants.keys():
            if key not in defaultSysConstants.keys():
                raise ValueError('The sysConstants key is not in the default sysConstants')
            defaultSysConstants[key]=sysConstants[key]
        if defaultSysConstants['Dissipation']==None:
            defaultSysConstants.pop('Dissipation')
        self.sysConstants=defaultSysConstants
        #print(self.sysConstants)
        return self.__forward_fn

    @torch.no_grad()
    def __predict_fun(self,input:torch.Tensor,net:StandardSNN,\
                        numPreds:int=1,multiPred:bool=True):
        '''
        name: predict_fun
        function: predict the next numPreds
        param {prefix}: the prefix of multi-prediction or single prediction input
        param {numPreds}: the number of multi-prediction, this argument is useless in single prediction
        param {net}: the network
        param {multiPred}: whether the prediction is multi-prediction
        return: the prediction
        '''
        state=net.begin_state(batch_size=1)
        outputs=[pre for pre in input[0:self.interval]]
        get_input=lambda: torch.unsqueeze(outputs[-self.interval],dim=0)
        if multiPred:
            #warm-up
            for Y in input[self.interval:]:
                _,state=net(get_input(),state)
                outputs.append(Y)
            for _ in range(numPreds):
                Y,state=net(get_input(),state)
                outputs.append(Y)
        else:
            for X in input:
                Y,state=net(X,state)
                outputs.append(Y)
        return self.outputTransoform(outputs)

    def get_predict_fun(self,outputTransoform:Callable=lambda x:x,\
                        interval:int=1):
        '''
        name: get_predict_fun
        fuction: get the function for prediction
        param{outputTransoform}: the transform function for output
        param{interval}: the interval of prediction
        return: the function
        '''        
        self.interval=interval
        self.outputTransoform=outputTransoform
        return self.__predict_fun

    @staticmethod
    def grad_clipping(net, theta):
        '''
        name: grad_clipping
        function: clip the gradients
        param {net}: the network
        param {theta}: the threshold
        '''
        params = [p for p in net.params if p.requires_grad]
        for param in params:
            param.grad.data.clamp_(-theta, theta)

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
            y=Y.transpose(0,1).reshape(-1,Y.shape[-1])
            y_hat,state=net(X,state)
            assert y_hat.shape==y.shape, 'y_hat.shape={}, y.shape={}'.format(y_hat.shape,y.shape)
            l=loss(y_hat,y).mean()
            if isinstance(updater,torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                QuantumSystemFunction.grad_clipping(net,clipTheta)
                updater.step()
            else:
                l.backward()
                QuantumSystemFunction.grad_clipping(net,clipTheta)
                updater(batch_size=1)
            metric.add(l*y.numel(),y.numel())
        return metric[0]/metric[1], metric[1]/timer.stop()

    @staticmethod
    @torch.no_grad()
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
            y=Y.transpose(0,1).reshape(-1,Y.shape[-1])
            y_hat,state=net(X,state)
            assert y_hat.shape==y.shape, 'y_hat.shape={}, y.shape={}'.format(y_hat.shape,y.shape)
            l=loss(y_hat,y).mean()
            metric.add(l*y.numel(),y.numel())
        return metric[0]/metric[1]
          
