import pandas as pd 
import inspect
import numpy as np

def parameters_to_pandas(method_init):

    '''
    Gets Parameters from Method. 
    Attributes: 
        method_init func: Class Instantiation to get Parameters from.
    Returns: 
        pd.DataFrame: New Dataframe with Current method parameters
    '''

    col=[]
    val=[]
    #print( method_init.__dict__.keys())
    for i in method_init.__dict__.keys():

        if i not in ['model', 'mode','device', 'NumTimeSteps', 'NumFeatures','backend','verbose','neighbors','change']:
            #'Grad','data','x','y','test_x'
            if '[' not in str(str(method_init.__dict__[i])):
                 if '>' not in str(str(method_init.__dict__[i])):
                    if '(' not in str(str(method_init.__dict__[i])):
                        #print(i)
                        #print(str(method_init.__dict__[i]))
                        col.append(i)
                        val.append(str(method_init.__dict__[i]))    
    return pd.DataFrame([val],columns=col)

def new_kwargs(method):
    '''
    Gets Parameters from Method. 
    Attributes: 
        method func: Class Instantiation to get Parameters from.
    Returns: 
        dict: New Dict with Current method parameters
    '''
    parameter_dict= method.__dict__
    di={}
    args= inspect.getfullargspec(type(method).__init__).args
    for a in parameter_dict.keys(): 
        if a in args:
            di[a]=parameter_dict[a]
    return di

class Quantus_Wrapper():
    def __init__(self,explainer,mode=None) :
        self.explainer=explainer
        self.mode=mode
    def make_callable(self,model,inputs,targets, device='cpu'):
        self.explainer.model=model
        res=[]
        print(targets)
        for x1, y1 in zip(inputs,targets):

            x1=np.array(x1)
            print('X1 ', x1.shape)
            print('y1 ', y1)

            if self.mode == 'time':
                a= np.array(self.explainer.explain(x1.reshape(-1,inputs.shape[-1],inputs.shape[-2]), int(y1)))
                if len(a)==2:
                    a,b=self.explainer.explain(x1.reshape(-1,inputs.shape[-1],inputs.shape[-2]), int(y1))

                a=np.array(a).reshape(-1,inputs.shape[-2],inputs.shape[-1])
                res.append(a)
            else:
                a= np.array(self.explainer.explain(x1.reshape(-1,inputs.shape[-2],inputs.shape[-1]), int(y1)))
                if len(a)==2:
                    a,b=self.explainer.explain(x1.reshape(-1,inputs.shape[-2],inputs.shape[-1]), int(y1))
                res.append(np.array( a))
        res=np.array(res)
        return np.array(res).reshape(-1, res.shape[-2], res.shape[-1])
    

#class Quantus_Wrapper():

#    def __init__(self,explainer) :
#        self.explainer=explainer
#    def make_callable(self,model,inputs,targets, device='cpu'):
 #       self.explainer.model=model
 #       res=[]
 #       for x1, y1 in zip(inputs,targets):
 #           res.append(self.explainer.explain(x1.reshape(-1,inputs.shape[-2],inputs.shape[-1]), int(y1)))
 #       #print('res',np.array(res).shape)               
 #       return res