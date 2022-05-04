'''
Name: CExpFN
Desriptption: Full power default SRNN
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: Experiment One
Author: YesunHuang
Date: 2022-04-17 20:40:50
'''
#import all the things we need
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import torch
def transform(Xs):
        return [torch.squeeze(x) for x in Xs]
#Some constants
GENERATE_DATA=False
TRAIN_NETWORK=True
SAVE_NETWORK=True
LOAD_NETWORK=False
PREDICTION_TEST=True

if __name__=='__main__':
    from DataGenerator.HenonMapDataGen import HenonMapDataGen
    from ClassicalModels.ClassicalSRNNs import ClassicalSRNN,SuportFunction
    from GradientFreeOptimizers.CostFunc import GradFreeMSELoss
    from GradientFreeOptimizers.Optimizers import MCSOptimizer
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
    filename='QExp1.csv'

    ## Generate Data
if GENERATE_DATA and __name__=='__main__':
    hmap=HenonMapDataGen(savepath=dataSavepath)
    hmap(1000)
    hmap.save_to_CSV(filename)

if __name__=='__main__':   
    ## Read the data
    hmap=HenonMapDataGen(savepath=dataSavepath)
    hmap.read_from_CSV(filename)
    ## Get the Iter
    trainIter,testIter=hmap.get_data_iter(testSetRatio,numStep,batchSize,mask=0,shuffle=False)
    
    ## Print information
if __name__=='__main__':
    print(hmap)
    X,Y=next(iter(trainIter))
    print('Train Data Size:',len(trainIter))
    X,Y=next(iter(testIter))
    print('Test Data Size:',len(testIter))

    # Load the network
if LOAD_NETWORK and __name__=='__main__':
    filename='ExpFN.pt'
    netData=torch.load(os.path.join(netSavepath,filename))

    inputSize=netData['inputSize']
    outputSize=netData['outputSize']
    hiddenSize=netData['hiddenSize']
    
    inputRatio=netData['inputRatio']
    outputRatio=netData['outputRatio']
    initValue=netData['initValue']
    
    inactive=netData['inactive']
    rescale=netData['rescale']

    isTypical=netData['isTypical']

elif __name__=='__main__':
    # Model
    ## Parameters
    inputSize=outputSize=1
    hiddenSize=2
    initValue=0
    inputRatio=outputRatio=1.0
    rescale=0.01
    inactive=[]
    isTypical=True

if __name__=='__main__':
    ## print parameters
    print('Input Ratio:',inputRatio)
    print('Output Ratio:',outputRatio)

if __name__=='__main__':
    ## Get neccesary functions
    srnnTestSup=SuportFunction()
    #transform=lambda Xs:[torch.squeeze(x) for x in Xs]
    init_rnn_state=srnnTestSup.get_init_state_fun(initStateValue=initValue)
    get_params=srnnTestSup.get_get_params_fun(inputRatio=inputRatio,\
                                                outputRatio=outputRatio,\
                                                rescale=rescale,\
                                                inactive=inactive)
    rnn=srnnTestSup.get_forward_fn_fun(isTypical=isTypical)
    predict_fun=srnnTestSup.get_predict_fun(outputTransoform=transform)

    net=ClassicalSRNN(inputSize,hiddenSize,outputSize,get_params,init_rnn_state,rnn)

if LOAD_NETWORK and __name__=='__main__':
    net.params=netData['NetParams']
    net.constants=netData['NetConstants']

## Test prediction
if __name__=='__main__':
    state=net.begin_state(batchSize)
    Y,newState=net(X,state)
    print(Y.shape, len(newState), newState[0][0].shape)

if not LOAD_NETWORK and not TRAIN_NETWORK:
    print('The network is not trained, are you sure to move on?')

    # Train the network
if  TRAIN_NETWORK and __name__=='__main__': 
    ## Parameters
    if LOAD_NETWORK:
        print('Are you sure to train the trained network?')
        num_epochs=netData['OptimizerConstant']['num_epochs']
        maxLevyStepSize=netData['OptimizerConstant']['maxLevyStepSize']
        regular=netData['OptimizerConstant']['regular']
        nestNum=netData['OptimizerConstant']['nestNum']
    else:
        num_epochs= 100
        maxLevyStepSize=[0.2]*5
        regular=None
        nestNum=40
    step_epochs=5

## Initial loss
if __name__=='__main__':
    ## Loss function
    lossFunc=GradFreeMSELoss(net)
    if LOAD_NETWORK:
        l_epochs=netData['Loss']
        print(f'Saved Train Loss: {l_epochs[-1][0]:f}')
        print(f'Saved Test Loss: {l_epochs[-1][1]:f}')
    else:
        l_epochs=[]
    timer=hp.Timer()
    train_l=SuportFunction.evaluate_accuracy(net,trainIter,lossFunc,False)
    t1=timer.stop()
    timer.start()
    test_l=SuportFunction.evaluate_accuracy(net,testIter,lossFunc,False)
    t2=timer.stop()
    l_epochs.append([train_l,test_l])
    print(f'Initial Train Loss: {train_l:f}, Time Cost: {t1:f}s')
    print(f'Initial Test Loss: {test_l:f}, Time Cost: {t2:f}s')
    
    ## Training
if TRAIN_NETWORK and __name__=='__main__':
    ## Optimizer
    mcs=MCSOptimizer(net.params,lossFunc,trainIter,nestNum=nestNum,\
        maxLevyStepSize=maxLevyStepSize,regular=regular,\
        randInit=True,epochToGeneration=lambda x:max(int(x/100),1))
    ## prediction
    predict = lambda prefix: predict_fun(prefix,net, numPreds=9)
    ## train and predict
    timer=hp.Timer()
    for epoch in range(num_epochs):
        trainLoss, _=mcs.step()
        testLoss=SuportFunction.evaluate_accuracy(net, testIter, lossFunc, False)
        if (epoch + 1) % step_epochs == 0:
            timeEpoch=timer.stop()
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {trainLoss:.4f}, Test Loss: {testLoss:.4f},\
                 Time: {timeEpoch:.4f}s') 
            timer.start()
        l_epochs.append([trainLoss,testLoss])
        #scheduler.step()
    testLoss=SuportFunction.evaluate_accuracy(net, testIter, lossFunc, False)
    print(f'TestLoss {testLoss:f}')

    ## Save the network
if SAVE_NETWORK and __name__=='__main__':
    ## Parameters
    filename='ExpFN.pt'
    OptimizerConstant={'num_epochs':num_epochs,'maxLevyStepSize':maxLevyStepSize,\
        'nestNum':nestNum}
    netData={'NetParams':net.params,'NetConstants':net.constants,\
            'inputSize':inputSize,'hiddenSize':hiddenSize,'outputSize':outputSize,\
            'inputRatio':inputRatio,'outputRatio':outputRatio,'initValue':initValue,\
            'inactive':inactive,'rescale':rescale,'isTypical':isTypical,\
            'Loss':l_epochs,'OptimizerConstant':OptimizerConstant}
    torch.save(netData,os.path.join(netSavepath,filename))

if PREDICTION_TEST and __name__=='__main__':
    # Prediction
    ## One-step prediction
    X,Y=next(iter(testIter))
    state=net.begin_state(batchSize)
    Y_hat,newState=net(X,state)
    Y=Y.transpose(0,1).reshape([-1,Y.shape[-1]])

    axes,fig=plt.subplots(1,1,figsize=(4,3))
    plt.title('One-Step Prediction')
    plt.plot(torch.linspace(1,Y.numel(),Y.numel()),torch.squeeze(Y),label='Y')
    plt.plot(torch.linspace(1,Y.numel(),Y.numel()),torch.squeeze(Y_hat).detach(),label=r'$\hat{Y}$')
    plt.legend()
    plt.show()

    ## Multi Step Prediction
    prefixSize=10
    totalSize=20
    testShift=int(len(hmap)*(1-testSetRatio))
    preX,preY=hmap.data_as_tensor
    preX,preY=torch.unsqueeze(preX[testShift:testShift+prefixSize],-1),torch.unsqueeze(preY[testShift:testShift+totalSize-1],-1)
    preY=[y for y in torch.cat((preX[:2],preY[1:]),dim=0)]
    preX=torch.unsqueeze(preX,-1)
    YHat=predict_fun(preX,net,numPreds=totalSize-prefixSize)

    axes,fig=plt.subplots(1,1,figsize=(4,3))
    plt.title('Multi-Step Prediction')
    fig.set_ylim(-2,2)
    plt.plot(torch.linspace(1,len(preY),len(preY)),preY,label='Y')
    plt.plot(torch.linspace(1,len(preY),len(preY)),YHat,label=r'$\hat{Y}$')
    plt.vlines([prefixSize-1],ymin=-2,ymax=2,linestyles='dashed',label='Prediction')
    plt.legend()
    plt.show()



