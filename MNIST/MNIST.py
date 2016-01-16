import numpy as np
import pandas as pd


# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("./Data/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("./Data/test.csv").values

# Initialize the parameters
m = target.shape[0]
n = train.shape[1]
hidden_layers = 3
hidden_units = np.array([800,800,800])
iter = 5
        
# Helper Functions

def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def plot_first_k_numbers(X,k):
    from matplotlib import pyplot
    m=X.shape[0]
    k=min(m,k)
    j = int(round(k / 10.0))
   
     
    fig, ax = pyplot.subplots(k,10)
    
    
    for i in range(k):
 
        w=X[i,:]
        
         
        w = w.reshape(28,28)
        ax[i/10,i%10].imshow(w,cmap=pyplot.cm.gist_yarg,
                      interpolation='nearest', aspect='equal')
        ax[i/10,i%10].axis('off')
 
     
    pyplot.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    pyplot.tick_params(\
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off', 
        right='off',    # ticks along the top edge are off
        labelleft='off')
      
    pyplot.show()
    
    
plot_first_k_numbers(train,100)    

import num