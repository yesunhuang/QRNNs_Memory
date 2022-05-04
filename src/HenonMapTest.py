
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from DataGenerator.HenonMapDataGen import HenonMapDataGen
#Some constants
GENERATE_DATA=False
FILE_NAME='QExp1.csv'
INTERVAL=1
OFFSET=1
TIME_PLOT_SIZE=100
PHASE_PLOT_SIZE=500

#set the save path
currentPath=os.getcwd()
dataSavepath=os.path.join(currentPath,'data','HenonMap','Exp')
figSavepath=os.path.join(currentPath,'data','figures')

# Generate Data
if GENERATE_DATA:
    hmap=HenonMapDataGen(savepath=dataSavepath,n=INTERVAL,heavyMem=False)
    hmap(1000)
    hmap.save_to_CSV(FILE_NAME)

# Read the data
hmap=HenonMapDataGen(savepath=dataSavepath)
hmap.read_from_CSV(FILE_NAME)
print(hmap)

#Plot the data
X,Y=hmap.data_as_array

#Plot the time series
Yt=Y[OFFSET:OFFSET+TIME_PLOT_SIZE]
fig,axes=plt.subplots(1,1,figsize=(4,3))
axes.plot(np.linspace(1,len(Yt),len(Yt)),Yt)
axes.set_ylabel(r'$y_t$')
axes.set_xlabel(r'$t$')
axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.show()
fig.savefig(os.path.join(figSavepath,'HenonT.svg'),dpi=600,format='svg',bbox_inches='tight')
fig.savefig(os.path.join(figSavepath,'HenonT.pdf'),dpi=600,format='pdf',bbox_inches='tight')

#Plot the phase space
Xp=X[OFFSET:OFFSET+PHASE_PLOT_SIZE]
Yp=Y[OFFSET:OFFSET+PHASE_PLOT_SIZE]
fig,axes=plt.subplots(1,1,figsize=(4,3))
axes.scatter(Xp,Yp,s=2.0)
axes.set_ylabel(r'$y_t$')
axes.set_xlabel(r'$x_t$')
plt.show()
fig.savefig(os.path.join(figSavepath,'HenonP.svg'),dpi=600,format='svg',bbox_inches='tight')
fig.savefig(os.path.join(figSavepath,'HenonP.pdf'),dpi=600,format='pdf',bbox_inches='tight')



