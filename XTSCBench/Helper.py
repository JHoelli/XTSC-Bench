
import numpy as np
from sklearn import preprocessing as pre
def counterfactual_manipulator(exp,data, meta, data_shape_1,data_shape_2,scaler, raw_data, scaling=True, labels=None,cf_man=True):
    
    exp=np.array(exp).reshape(-1,data_shape_1*data_shape_2)

    tmp = np.where(np.isin(exp, [None]), np.nan,exp).astype(float)
    tmp_index=np.where(~np.isnan(tmp).all(axis=1))
    #print(tmp_index)

    if len(tmp_index[0]) > 1:
        data=data[tmp_index[0]]
        if meta is not None:
            meta=meta[tmp_index[0]]
        if labels is not None : 
            labels=labels[tmp_index[0]]
        if raw_data is not None:
            raw_data=raw_data[tmp_index[0]]
    else:
        return [],[],[],[]
    


    new_arr = tmp[~np.isnan(tmp).all(axis=1)]
    exp = new_arr
    if not scaling:
        return exp, data, meta,labels

    if raw_data is not None:
        raw_data_scaled = scaler.transform(raw_data.reshape(-1,data_shape_1*data_shape_2))
    if cf_man:                        
        exp_man = exp.reshape(-1,data_shape_1*data_shape_2)-raw_data_scaled.reshape(-1,data_shape_1*data_shape_2)
        exp_man = pre.MinMaxScaler().fit_transform(exp_man)
        exp=exp_man.reshape(-1,data_shape_1,data_shape_2)
    return exp, data, meta,labels