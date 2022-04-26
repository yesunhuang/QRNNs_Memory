'''
Name: StandardTest
Desriptption: Performing the standard test for QRNN
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: Experiment
Author: YesunHuang
Date: 2022-04-17 20:40:50
'''
#import all the things we need
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import torch
def transform(Xs):
        return [torch.squeeze(x) for x in Xs]
#Some constants
##File names
DATA_FILENAME='QExp1.csv'
NET_FILENAME='QExp1.pt'
##Loss test configuration
TRIALS=10
TRIALS_STEP=1
TEST_TRIAN_DATA=False
##Prediction plot configuration
MULTI_PREFIX_SIZE=10
MULTI_TOTAL_SIZE=20
SINGLE_TOTAL_SIZE=20
DATA_SHIFT=0


if __name__=='__main__':
    from DataGenerator.HenonMapDataGen import HenonMapDataGen
    from QuantumModels.QuantumSRNNs import QuantumSRNN
    from QuantumModels.QuantumSRNNs import QuantumSystemFunction
    from GradientFreeOptimizers.CostFunc import GradFreeMSELoss
    import GradientFreeOptimizers.Helpers as hp

    #Save path:
if __name__=='__main__':
    currentPath=os.getcwd()
    dataSavepath=os.path.join(currentPath,'data','HenonMap','Exp')
    netSavepath=os.path.join(currentPath,'TrainedNet','Exp')

if __name__=='__main__':
    # Data Iter
    ## Parameters
    testSetRatio=0.2
    numStep=10
    batchSize=16

if __name__=='__main__':   
    ## Read the data
    hmap=HenonMapDataGen(savepath=dataSavepath)
    hmap.read_from_CSV(DATA_FILENAME)
    ## Get the Iter
    trainIter,testIter=hmap.get_data_iter(testSetRatio,\
        numStep,batchSize,mask=0,shuffle=False,randomOffset=False)
    
    ## Print information
if __name__=='__main__':
    print(hmap)
    X,Y=next(iter(trainIter))
    print('Train Data Size:',len(trainIter))
    X,Y=next(iter(testIter))
    print('Test Data Size:',len(testIter))

    # Load the network
if  __name__=='__main__':
    netData=torch.load(os.path.join(netSavepath,NET_FILENAME))

    inputSize=netData['inputSize']
    outputSize=netData['outputSize']
    qubits=netData['qubits']
    
    inputQubits=netData['inputQubits']
    outputQubits=netData['outputQubits']

    isDensity=netData['isDensity']
    activation=netData['activation']
    
    interQPairs=netData['interQPairs']
    inactive=netData['inactive']
    rescale=netData['rescale']

    sysConstants=netData['sysConstants']
    measEffect=netData['measEffect']  
    if isDensity:
        samples=netData['samples']
    else:    
        samples=1

    sysConstants['numCpus']=1

if __name__=='__main__':
    ## print parameters
    if isDensity:
        print('QRNN Type: Unlimited Samples')
    else:
        print(f'QRNN Type: Limited Samples of {samples:d}')
    if measEffect:
        print('Measurement Effect is enabled.')
    else:
        print('Measurement Effect is disabled.')
    print('Net Configuration:')
    print('-'*50)
    print('Total Qubits:',qubits)
    print('\tInput Qubits:\n',inputQubits)
    print('\tOutput Qubits:\n',outputQubits)
    print('\tInitial Activation:\n',activation)
    print('\tInteraction Pairs:\n',interQPairs)
    print('\tSysConstant=\n',sysConstants)
    print('-'*50)
    

if __name__=='__main__':
    ## Get neccesary functions
    srnnTestSup=QuantumSystemFunction()
    #transform=lambda Xs:[torch.squeeze(x) for x in Xs]
    init_rnn_state=srnnTestSup.get_init_state_fun(activation=activation,\
                                                isDensity=isDensity)
    get_params=srnnTestSup.get_get_params_fun(inputQubits=inputQubits,\
                                            outputQubits=outputQubits,\
                                            interQPairs=interQPairs,\
                                            inactive=inactive,\
                                            rescale=rescale)
    rnn=srnnTestSup.get_forward_fn_fun(samples=samples,\
                                        measEffect=measEffect,\
                                        sysConstants=sysConstants)
    predict_fun=srnnTestSup.get_predict_fun(outputTransoform=transform)

    net=QuantumSRNN(inputSize,qubits,outputSize,get_params,init_rnn_state,rnn)

if  __name__=='__main__':
    net.params=netData['NetParams']
    net.constants=netData['NetConstants']

## Loss
if __name__=='__main__':
    ## Loss function
    lossFunc=GradFreeMSELoss(net)
    l_epochs=netData['Loss']
    print(f'Saved Train Loss: {l_epochs[-1][0]:f}')
    print(f'Saved Test Loss: {l_epochs[-1][1]:f}')

    timer=hp.Timer()
    train_loss=[]
    test_loss=[]
    for i in range(len(TRIALS)):
        test_loss.append(QuantumSystemFunction.evaluate_accuracy(net, testIter, lossFunc, True))
        if TEST_TRIAN_DATA:
            train_loss.append(QuantumSystemFunction.evaluate_accuracy(net, trainIter, lossFunc, True))
        if (i+1)%TRIALS_STEP==0:
            timeCost=timer.stop()
            print('-'*50)
            print(f'Trial {i+1:d}/{TRIALS[-1]:d}')
            print(f'Test Loss: {test_loss[-1]:f}')
            if TEST_TRIAN_DATA:
                print(f'Train Loss: {train_loss[-1]:f}')
            print(f'Time Cost: {timeCost:f}s')
            timer.start()
    print('-'*50)
    print(f'Average Test Loss: {np.mean(test_loss):f}')
    if TEST_TRIAN_DATA:
        print(f'Average Train Loss: {np.mean(train_loss):f}')
    

if  __name__=='__main__':
    # Prediction
    ## One-step prediction
    testShift=int(len(hmap)*(1-testSetRatio))+DATA_SHIFT
    preX,preY=hmap.data_as_tensor
    preX,preY=torch.unsqueeze(preX[testShift:testShift+SINGLE_TOTAL_SIZE],-1),\
        torch.unsqueeze(preY[testShift:testShift+SINGLE_TOTAL_SIZE-1],-1)
    preY=[y for y in torch.cat((preX[:2],preY[1:]),dim=0)]
    preX=torch.unsqueeze(preX,-1)
    YHat=predict_fun(preX,net,multiPred=False)

    axes,fig=plt.subplots(1,1,figsize=(4,3))
    plt.title('Single-Step Prediction')
    fig.set_ylim(-2,2)
    plt.plot(torch.linspace(0,len(preY),len(preY)),preY,label='Y')
    plt.plot(torch.linspace(0,len(preY),len(preY)),YHat,label=r'$\hat{Y}$')
    plt.vlines([MULTI_PREFIX_SIZE-1],ymin=-2,ymax=2,linestyles='dashed',label='Prediction')
    plt.legend()
    plt.show()

if __name__=='__main__':
    ## Multi Step Prediction
    testShift=int(len(hmap)*(1-testSetRatio))+DATA_SHIFT
    preX,preY=hmap.data_as_tensor
    preX,preY=torch.unsqueeze(preX[testShift:testShift+MULTI_PREFIX_SIZE],-1),\
        torch.unsqueeze(preY[testShift:testShift+MULTI_TOTAL_SIZE-1],-1)
    preY=[y for y in torch.cat((preX[:2],preY[1:]),dim=0)]
    preX=torch.unsqueeze(preX,-1)
    YHat=predict_fun(preX,net,numPreds=MULTI_TOTAL_SIZE-MULTI_PREFIX_SIZE)

    axes,fig=plt.subplots(1,1,figsize=(4,3))
    plt.title('Multi-Step Prediction')
    fig.set_ylim(-2,2)
    plt.plot(torch.linspace(0,len(preY),len(preY)),preY,label='Y')
    plt.plot(torch.linspace(0,len(preY),len(preY)),YHat,label=r'$\hat{Y}$')
    plt.vlines([MULTI_PREFIX_SIZE-1],ymin=-2,ymax=2,linestyles='dashed',label='Prediction')
    plt.legend()
    plt.show()



