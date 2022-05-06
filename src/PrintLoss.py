'''
Name: QExp1
Desriptption: Experiment One 
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: Experiment One
Author: YesunHuang
Date: 2022-04-17 20:40:50
'''
#import all the things we need
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import matplotlib.pyplot as plt

DRAW_LOSS=True
    #Save path:
if __name__=='__main__':
    currentPath=os.getcwd()
    netSavepath=os.path.join(currentPath,'TrainedNet','Exp')
    figSavepath=os.path.join(currentPath,'data','figures')

if __name__=='__main__':
    QFPM_filename='QExpF1.pt'
    QFPM_FT_filename='QExpFT1.pt'
    CSM_filename='CExpFN.pt'
    CQCM_filename='CExpFQC.pt'
    QISMM_filename='QExpFM.pt'
    QFPMD_filename='QExpFD.pt'

# Load loss
if  __name__=='__main__':
    loss_QFPM=torch.load(os.path.join(netSavepath,QFPM_filename))['Loss']
    trainLoss_QFPM=[l[0] for l in loss_QFPM];testLoss_QFPM=[l[1] for l in loss_QFPM]
    loss_QFPM_FT=torch.load(os.path.join(netSavepath,QFPM_FT_filename))['Loss']
    trainLoss_QFPM_FT=[l[0] for l in loss_QFPM_FT];testLoss_QFPM_FT=[l[1] for l in loss_QFPM_FT]
    loss_CSM=torch.load(os.path.join(netSavepath,CSM_filename))['Loss']
    trainLoss_CSM=[l[0] for l in loss_CSM];testLoss_CSM=[l[1] for l in loss_CSM]
    loss_CQCM=torch.load(os.path.join(netSavepath,CQCM_filename))['Loss']
    trainLoss_CQCM=[l[0] for l in loss_CQCM];testLoss_CQCM=[l[1] for l in loss_CQCM]
    loss_QISMM=torch.load(os.path.join(netSavepath,QISMM_filename))['Loss']
    trainLoss_QISMM=[l[0] for l in loss_QISMM];testLoss_QISMM=[l[1] for l in loss_QISMM]
    loss_QFPMD=torch.load(os.path.join(netSavepath,QFPMD_filename))['Loss']
    trainLoss_QFPMD=[l[0] for l in loss_QFPMD];testLoss_QFPMD=[l[1] for l in loss_QFPMD]
    
# Draw loss
if DRAW_LOSS and __name__=='__main__':
    figName='TrainLoss'
    fig,axes=plt.subplots(1,1,figsize=(8,6))
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Train Loss')
    #axes.set_xlim(0,300)
    cor={'QFPM':'lightskyblue','QFPM_FT':'limegreen','CSM':'lightcoral',\
        'CQCM':'khaki','QISMM':'orange','QFPMD':'violet'}
    axes.plot(range(0,len(trainLoss_QFPM)),trainLoss_QFPM,color=cor['QFPM'],linestyle='-',label='QFPM')
    axes.plot(range(0,len(trainLoss_QFPMD)),trainLoss_QFPMD,color=cor['QFPMD'],linestyle='-',label='QFPMD')
    axes.plot(range(0,len(trainLoss_QISMM)),trainLoss_QISMM,color=cor['QISMM'],linestyle='-',label='QISMM')
    axes.plot(range(0,len(trainLoss_QFPM_FT[302:])),trainLoss_QFPM_FT[302:],color=cor['QFPM_FT'],linestyle='-',label='QFPM(FT)')
    axes.plot(range(0,len(trainLoss_CSM)),trainLoss_CSM,color=cor['CSM'],linestyle='-',label='CSM')
    axes.plot(range(0,len(trainLoss_CQCM)),trainLoss_CQCM,color=cor['CQCM'],linestyle='-',label='CQCM')
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(figSavepath,figName+'.svg'),dpi=600,format='svg',bbox_inches='tight')
    fig.savefig(os.path.join(figSavepath,figName+'.pdf'),dpi=600,format='pdf',bbox_inches='tight')