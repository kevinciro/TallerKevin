import pandas as pd
import torch

def getInitSSFromFile(cgmTarget,device='cpu'):
    if cgmTarget<40:
        cgmTarget = 40
    if cgmTarget>400:
        cgmTarget = 400
    dfInitStates = pd.read_csv('utils/initSteadyStates.csv')

    dfInitStates = dfInitStates.set_index('initCGM')
    x0 = dfInitStates.loc[int(cgmTarget),:]
    x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=False).to(device)
    return x0
