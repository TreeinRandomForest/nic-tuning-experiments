import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

plt.ion()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#TODO: pass train_df and test_df to create Dataset objects
class Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.N_cols = df.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        x = np.array(self.df.iloc[ix])
        
        #TODO: map features and labels appropriately
        features = x[:(self.N_cols-1)] 
        label = x[[-1]]

        return (torch.from_numpy(features).float(), torch.from_numpy(label))

#TODO: create an instance of Net with N_inputs = number of features, N_outputs = 2(time, energy), N_hidden_layers=1 or 2, N_hidden_nodes=128
#TODO: activation = nn.ReLU(), output_activation = None (TODO: maybe change output_activation to exp() to impose positivity)
class Net(nn.Module):
    def __init__(self, N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation):
        super(Net, self).__init__()
        
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        
        self.N_hidden_layers = N_hidden_layers
        self.N_hidden_nodes = N_hidden_nodes
        
        self.layer_list = nn.ModuleList([]) #use just as a python list
        for n in range(N_hidden_layers):
            if n==0:
                self.layer_list.append(nn.Linear(N_inputs, N_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(N_hidden_nodes, N_hidden_nodes))
        
        self.output_layer = nn.Linear(N_hidden_nodes, N_outputs)
        
        self.activation = activation
        self.output_activation = output_activation
        
    def forward(self, inp):
        out = inp
        for layer in self.layer_list:
            out = layer(out)
            out = self.activation(out)
            
        out = self.output_layer(out)
        if self.output_activation is not None:
            pred = self.output_activation(out)
        else:
            pred = out
        
        return pred


def train_model(train_dl, test_dl, model, criterion, N_epochs, print_freq, lr=1e-3, optimizer='adam'):
    '''Loop over dataset in batches, compute loss, backprop and update weights
    '''
    
    model.train() #switch to train model (for dropout, batch normalization etc.)
    
    model = model.to(device)
    if optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print("Using adam")
    elif optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print("Using sgd")
    else:
        raise ValueError("Please use either adam or sgd")
    
    loss_dict = {}
    for epoch in range(N_epochs): #loop over epochs i.e. sweeps over full data
        curr_loss = 0
        N = 0
        
        for idx, (features, labels) in enumerate(train_dl): #loop over batches = random samples from train dataset
            #move features and labels to GPU if needed
            features = features.to(device)
            labels = labels.to(device)
            
            preds = model(features) #make predictions
            loss = criterion(preds.squeeze(), labels.squeeze().float()) #compute loss between predictions and labels
            
            curr_loss += loss.item() #accumulate loss
            N += len(labels) #accumulate number of data points seen in this epoch
                
            #backprop and updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % print_freq == 0 or epoch==N_epochs-1:
            val_loss = validate(test_dl, model, criterion) #get model perf metrics from test set
            
            loss_dict[epoch] = val_loss
            
            print(f'Iter = {epoch} Train Loss = {curr_loss / N} val_loss = {val_loss}')
            
    return model, loss_dict

def validate(test_dl, model, criterion):
    '''Loop over test dataset and compute loss and accuracy
    '''
    model.eval() #switch to eval model
    
    loss = 0
    N = 0

    preds_all, labels_all = torch.tensor([]), torch.tensor([])
    
    with torch.no_grad(): #no need to keep variables for backprop computations
        for idx, (features, labels) in enumerate(test_dl): #loop over batches from test set
            features = features.to(device)
            labels = labels.to(device).float()
            
            preds = model(features)
            
            preds_all = torch.cat((preds_all, preds.to('cpu')), 0)
            labels_all = torch.cat((labels_all, labels.to('cpu')), 0)
            
            loss += criterion(preds.squeeze(), labels.squeeze()) #cumulative loss
            N += len(labels)
    
    #avg_precision = average_precision_score(labels_all.squeeze().numpy(), preds_all.squeeze().numpy())
    
    return loss / N

def run():
    df = ... #TODO: read csv file

    #Split into train-test randomly
    df_train, df_test = train_test_split(df, train_size=0.7)

    #Create Dataset objects
    ds_torch_train = Dataset(df_train)
    ds_torch_test = Dataset(df_test)

    #Create Dataloader objects (to sample batches of rows)
    batch_size = 32
    dl_torch_train = DataLoader(ds_torch_train, batch_size=batch_size, num_workers=0)
    dl_torch_test = DataLoader(ds_torch_test, batch_size=batch_size, num_workers=0)

    #criterion i.e. loss function
    criterion = nn.MSELoss()

    #init and train model
    model = Net(...) #TODO: appropriate arguments
    model, loss_dict = train_model(...) #TODO: arguments