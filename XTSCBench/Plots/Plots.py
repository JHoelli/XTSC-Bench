
import pandas as pd 
import numpy as  np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import os
import plotly.graph_objects as go
from Benchmarking.metrics.synthetic_helper import *

def acc_plot(path,metric_name='acc',split_column1='params_exp',grouping=None,split_column2='info_feat',figsize=None,acc_model_threshold= None, with_acc=True,save_path=None, methods=['FO','GRAD','GS','SG','FO - TSR','GRAD - TSR','GS - TSR','SG - TSR','TSEvo - authentic_opposing_information','LEFTIST - Lime' ,'NativeGuideCF - NUN_CF','TSEvo']):
        '''
        #TODO This needs to be implemented
        Every Infitmszive Feature , get an own graph

        '''
        data_full=None
        for file in os.listdir(path):
            print(file)
            #TODO ELIMINATE INDES ETC
            data_temp = pd.read_csv(f'{path}/{file}',index_col=0)
            if 'CNN' in file:
                temp_column = np.repeat('CNN',len(data_temp))
            elif 'LSTM' in file:
                temp_column = np.repeat('LSTM',len(data_temp))
            data_temp['model']=temp_column

            temp_name = np.repeat(file.split('.npy')[0]+'.npy',len(data_temp))
            data_temp['name']=temp_name
            data_temp['name']=data_temp['name'].str.replace('Testing','Training')

            temp_info= file.split('_')
    
            gen=temp_info[2]
            info_feat=temp_info[1]
 
            if  info_feat== 'Moving':
                info_feat=temp_info[1]+ temp_info[2]
                gen=temp_info[3]
            params= file.split('[')[-1].replace('.csv','')
            data_temp['generation']= np.repeat(gen, len(data_temp))
      
            data_temp['info_feat']= np.repeat(info_feat, len(data_temp))
            if 'univariate' in path:
                data_temp=data_temp[data_temp['info_feat']!='RareFeature'] 

            params=params.replace('[','')
            params=params.replace(']','')
            params=params.replace('\'','')
        
            try:
                params_exp=data_temp['explanation'][0].split('\'')[0]
            except: 
                continue
            if 'Saliency' in params_exp:
                params_exp=params_exp.replace('_PTY','')
                print(params_exp)
                sp =  params.split(' ')
                print(sp)
                if sp[-1]=='False':
                    params_exp =sp[0]
                else: 
                    params_exp = sp[0]+ ' - TSR'

               
            elif 'LEFTIST' in params_exp:
                sp =  params.split(' ')
                params_exp = params_exp+' - '  +sp[3]
            elif 'TSEvo' in params_exp:
                sp =  params.split(' ')
                if 'authentic'in sp[0]:
                    pass
                else:
                    params_exp = params_exp+' - '  +sp[0]
            elif 'NativeGuide' in params_exp:
                #print('paramts',params)
                sp =  params.split(' ')
               # print(sp)
                params_exp = params_exp+' - '  +sp[1]


            data_temp['params']= np.repeat(params, len(data_temp))
            data_temp['params_exp']= np.repeat(params_exp, len(data_temp))

            if methods is not None:
                data_temp = data_temp[data_temp['params_exp'].isin(methods)]
            array=[]
            perturb_array=['FO','FO - TSR','LEFTIST - Shap','LEFTIST - Lime' ]
            gradient_array=['GRAD','SG','GRAD - TSR','SG - TSR','GS','GS - TSR']
            example_array=['TSEvo','NativeGuideCF - NG','NativeGuideCF - dtw','NativeGuideCF - NUN_CF','TSEvo']
            for _,item in data_temp.iterrows():
                if item['params_exp'] in perturb_array :
                    array.append('Perturbation')
                if item['params_exp'] in gradient_array :
                    array.append('Gradient')
                if item['params_exp'] in example_array :
                    array.append('Example')
            
            data_temp['Based']= array
            data_temp['params_exp'] = data_temp['params_exp'].replace('NativeGuideCF - NUN_CF','NativeGuideCF')
            data_temp['params_exp'] = data_temp['params_exp'].replace('LEFTIST - Lime','LEFTIST')
                    

            if '_1_' in file: 
                acc=pd.read_csv('./Benchmarking/ClassificationModels/models_new/preformance_univariate.csv',index_col=0)
            else: 
                acc=pd.read_csv('./Benchmarking/ClassificationModels/models_new/preformance_multivariate.csv',index_col=0)

            new_df = pd.merge(data_temp, acc,  how='left', on=['name','model'],validate='m:1')

            data_temp=new_df

            flag = True

            if  len(data_temp['model'])==0:
                continue
            #if acc_model_threshold is not None: 
            #   
            temp= acc[acc['model']==data_temp['model'][0]]
            f=file.split('.npy')[0]+'.npy'
            f=f.replace('Testing','Training')
            temp = temp[temp['name']==f]#

            try: 
                if temp['acc'].values[0]> 0.9:
                    data_temp['accuracy']=np.repeat('acc>0.9',len(data_temp))
                elif temp['acc'].values[0]> 0.8 and temp['acc'].values[0]<=0.9:
                    data_temp['accuracy']=np.repeat('acc>0.8',len(data_temp))
                elif temp['acc'].values[0]> 0.7 and temp['acc'].values[0]<=0.8:
                    data_temp['accuracy']=np.repeat('acc>0.7',len(data_temp))
                elif temp['acc'].values[0]> 0.6 and temp['acc'].values[0]<=0.7:
                    data_temp['accuracy']=np.repeat('acc>0.6',len(data_temp))
                else: 
                    data_temp['accuracy']=np.repeat('nan',len(data_temp))
            except:
                pass 
            #        t=temp['model']#
            #
            #        print(f'MODEL DATA NOT FOUND FOR {t} {f} ')
            print(data_temp)
            if data_full is None: 
                data_full = data_temp
            elif flag: 
                data_full=pd.concat([data_full, data_temp],ignore_index=True)
        data_full=data_full.sort_values([f'{split_column1}',f'{split_column2}'])
        #data_full.to_csv('temp.csv')

 
        fig = go.Figure()
        
        i=0
        colors=['#D3D3D3','#ffffff','#545353','#858484']
        for m in ['acc>0.9','acc>0.8','acc>0.7','acc>0.6']:
            print(m)
            m_name= m.replace('_',' ')
            data_man=data_full[data_full['accuracy']==m]
            print(data_man)
            fig.add_trace(go.Box(x=[data_man[f'{split_column1}'],data_man[f'{split_column2}']], y=data_man[f'{metric_name}'], name=f'{m_name}',boxmean=True,fillcolor=colors[i], line={'color':'black'}) )

            if 'Sens' in m:
                fig.update_yaxes(range=[0, 2])
            if 'faith' in m:
                fig.update_yaxes(range=[-1, 1])
            if 'Rank' in m: 
                fig.update_yaxes(range=[0, 1])

            i+=1
        if 'rel' in metric_name:
            fig.update_yaxes(range=[0, 1])
        if 'faith' in metric_name:
            fig.update_yaxes(range=[-1, 1])
        if 'Rel' in metric_name: 
                fig.update_yaxes(range=[0, 1])

        fig.update_layout(
        yaxis_title='values',
        boxmode='group' # group together boxes of the different traces for each value of x
         )
        #width=500,
        #height=500,
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
    
        ),  
        margin=dict(l=0,r=0,b=0,t=0),
        template= 'plotly_white'
        )
        if figsize is not None: 
            #width=500,
        #height=500,
        
            fig.update_layout(
                width=figsize[0],
                height=figsize[1],
                font=dict(
  
        size=24,  # Set the font size here
       
    )
         
    
            )

        
        #fig.show()
        #import sys 
        #sys.exit(1)
        fig.write_image(f"{save_path}")
        fig.write_image(f"{save_path}.svg")
        return fig

def informative_feat_approach_box_plot(path,metric_name='acc',split_column1='params_exp',grouping=None,split_column2='info_feat',figsize=None,acc_model_threshold= None, with_acc=True,save_path=None, methods=['FO','GRAD','GS','SG','FO - TSR','GRAD - TSR','GS - TSR','SG - TSR','TSEvo - authentic_opposing_information','LEFTIST - Lime' ,'NativeGuideCF - NUN_CF','TSEvo']):
        '''
        #TODO This needs to be implemented
        Every Infitmszive Feature , get an own graph

        '''
        data_full=None
        for file in os.listdir(path):
            print(file)
            #TODO ELIMINATE INDES ETC
            data_temp = pd.read_csv(f'{path}/{file}',index_col=0)
            print(len(data_temp))
            if 'CNN' in file:
                temp_column = np.repeat('CNN',len(data_temp))
            elif 'LSTM' in file:
                temp_column = np.repeat('LSTM',len(data_temp))
            print(len(temp_column))
            data_temp['model']=temp_column

            temp_name = np.repeat(file.split('.npy')[0]+'.npy',len(data_temp))
            data_temp['name']=temp_name
            data_temp['name']=data_temp['name'].str.replace('Testing','Training')

            temp_info= file.split('_')
    
            gen=temp_info[2]
            info_feat=temp_info[1]
 
            if  info_feat== 'Moving':
                info_feat=temp_info[1]+ temp_info[2]
                gen=temp_info[3]
            params= file.split('[')[-1].replace('.csv','')
            data_temp['generation']= np.repeat(gen, len(data_temp))
      
            data_temp['info_feat']= np.repeat(info_feat, len(data_temp))
            if 'univariate' in path:
                data_temp=data_temp[data_temp['info_feat']!='RareFeature'] 

            params=params.replace('[','')
            params=params.replace(']','')
            params=params.replace('\'','')
        
            try:
                params_exp=data_temp['explanation'][0].split('\'')[0]
            except: 
                continue
            if 'Saliency' in params_exp:
                params_exp=params_exp.replace('_PTY','')
                print(params_exp)
                sp =  params.split(' ')
                print(sp)
                if sp[-1]=='False':
                    params_exp =sp[0]
                else: 
                    params_exp = sp[0]+ ' - TSR'

               
            elif 'LEFTIST' in params_exp:
                sp =  params.split(' ')
                params_exp = params_exp+' - '  +sp[3]
            elif 'TSEvo' in params_exp:
                sp =  params.split(' ')
                if 'authentic'in sp[0]:
                    pass
                else:
                    params_exp = params_exp+' - '  +sp[0]
            elif 'NativeGuide' in params_exp:
                #print('paramts',params)
                sp =  params.split(' ')
               # print(sp)
                params_exp = params_exp+' - '  +sp[1]


            data_temp['params']= np.repeat(params, len(data_temp))
            data_temp['params_exp']= np.repeat(params_exp, len(data_temp))

            if methods is not None:
                data_temp = data_temp[data_temp['params_exp'].isin(methods)]
            array=[]
            perturb_array=['FO','FO - TSR','LEFTIST - Shap','LEFTIST - Lime' ]
            gradient_array=['GRAD','SG','GRAD - TSR','SG - TSR','GS','GS - TSR']
            example_array=['TSEvo','NativeGuideCF - NG','NativeGuideCF - dtw','NativeGuideCF - NUN_CF','TSEvo']
            for _,item in data_temp.iterrows():
                if item['params_exp'] in perturb_array :
                    array.append('Perturbation')
                if item['params_exp'] in gradient_array :
                    array.append('Gradient')
                if item['params_exp'] in example_array :
                    array.append('Example')
            
            data_temp['Based']= array
            data_temp['params_exp'] = data_temp['params_exp'].replace('NativeGuideCF - NUN_CF','NativeGuideCF')
            data_temp['params_exp'] = data_temp['params_exp'].replace('LEFTIST - Lime','LEFTIST')
                    

            if '_1_' in file: 
                acc=pd.read_csv('./Benchmarking/ClassificationModels/models_new/preformance_univariate.csv',index_col=0)
            else: 
                acc=pd.read_csv('./Benchmarking/ClassificationModels/models_new/preformance_multivariate.csv',index_col=0)

            new_df = pd.merge(data_temp, acc,  how='left', on=['name','model'],validate='m:1')

            data_temp=new_df

            flag = True

            if  len(data_temp['model'])==0:
                continue
            if acc_model_threshold is not None: 
               
                temp= acc[acc['model']==data_temp['model'][0]]
                f=file.split('.npy')[0]+'.npy'
                f=f.replace('Testing','Training')
                temp = temp[temp['name']==f]

                try: 
                    flag=temp['acc'].values[0]> acc_model_threshold
                except: 
                    t=temp['model']

                    print(f'MODEL DATA NOT FOUND FOR {t} {f} ')

            if data_full is None and flag: 
                data_full = data_temp
            elif flag: 
                data_full=pd.concat([data_full, data_temp],ignore_index=True)
        data_full=data_full.sort_values([f'{split_column1}',f'{split_column2}'])
        #data_full.to_csv('temp.csv')

 
        fig = go.Figure()
        if grouping is not None:
            i=0
            colors=['#D3D3D3','#ffffff','#545353']
            for m in grouping:
                print(m)
                m_name= m.replace('_',' ')
                data_man=data_full[data_full['model']==m]
                print(data_man)
                fig.add_trace(go.Box(x=[data_man[f'{split_column1}'],data_man[f'{split_column2}']], y=data_man[f'{metric_name}'], name=f'{m_name}',boxmean=True,fillcolor=colors[i], line={'color':'black'}) )
                #fig.update_xaxes(range=[1.5, 4.5])
                #if 'sens' in m:
                #    fig.update_yaxes(range=[0, 1])
                if 'Sens' in m:
                    fig.update_yaxes(range=[0, 2])
                if 'faith' in m:
                    fig.update_yaxes(range=[-1, 1])
                if 'Rank' in m: 
                    fig.update_yaxes(range=[0, 1])

                i+=1
            
        elif type(metric_name)==str:
            fig.add_trace(go.Box(x=[data_full[f'{split_column1}'],data_full[f'{split_column2}']], y=data_full[f'{metric_name}'], name=f'{metric_name}',fillcolor='#D3D3D3', line={'color':'black'},boxmean=True) )
        else: 
            i=0
            colors=['#D3D3D3','#ffffff','#545353']
            for m in metric_name:
                print(m)
                m_name= m.replace('_',' ')
                fig.add_trace(go.Box(x=[data_full[f'{split_column1}'],data_full[f'{split_column2}']], y=data_full[f'{m}'], name=f'{m_name}',boxmean=True,fillcolor=colors[i], line={'color':'black'}) )
                #fig.update_xaxes(range=[1.5, 4.5])
                #if 'sens' in m:
                #    fig.update_yaxes(range=[0, 1])
                if 'Sens' in m:
                    fig.update_yaxes(range=[0, 2])
                if 'faith' in m:
                    fig.update_yaxes(range=[-1, 1])
                if 'Rank' in m: 
                    fig.update_yaxes(range=[0, 1])
                i+=1
        if with_acc:
            fig.add_trace(go.Box(x=[data_full[f'{split_column1}'],data_full[f'{split_column2}']], y=data_full[f'auc'],name='Model Acc',boxmean=True))
        #fig.add_trace(go.Scatter(x=data_temp[f'{split_column}'], y=data_temp['auc'], name="auc"))
        if 'rel' in metric_name:
            fig.update_yaxes(range=[0, 1])
        if 'faith' in metric_name:
            fig.update_yaxes(range=[-1, 1])
        if 'Rel' in metric_name: 
                fig.update_yaxes(range=[0, 1])

        fig.update_layout(
        yaxis_title='values',
        boxmode='group' # group together boxes of the different traces for each value of x
         )
        #width=500,
        #height=500,
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
    
        ),  
        margin=dict(l=0,r=0,b=0,t=0),
        template= 'plotly_white'
        )
        if figsize is not None: 
            #width=500,
        #height=500,
        
            fig.update_layout(
                width=figsize[0],
                height=figsize[1],
                font=dict(
  
        size=24,  # Set the font size here
       
    )
         
    
            )

        
        #fig.show()
        #import sys 
        #sys.exit(1)
        fig.write_image(f"{save_path}")
        fig.write_image(f"{save_path}.svg")
        return fig

def Latex_Table(elementwise_dir,acc_model_threshold=0.9,methods=['FO','GRAD','GS','SG','FO - TSR','GRAD - TSR','GS - TSR','SG - TSR','TSEvo - information','LEFTIST - Shap','LEFTIST - Lime' ,'NativeGuideCF - NG','NativeGuideCF - dtw','NativeGuideCF - NUN_CF']):
    strin=''
    for metric in os.listdir(elementwise_dir):
        data_full=None

        for file in os.listdir(f'{elementwise_dir}/{metric}'):
             # print(file)
            #TODO ELIMINATE INDES ETC
            data_temp = pd.read_csv(f'{elementwise_dir}/{metric}/{file}',index_col=0)
            if 'CNN' in file:
                temp_column = np.repeat('CNN',len(data_temp))
            elif 'LSTM' in file:
                temp_column = np.repeat('LSTM',len(data_temp))
            data_temp['model']=temp_column

            temp_name = np.repeat(file.split('.npy')[0]+'.npy',len(data_temp))
            data_temp['name']=temp_name
            data_temp['name']=data_temp['name'].str.replace('Testing','Training')

            temp_info= file.split('_')
    
            gen=temp_info[2]
            info_feat=temp_info[1]
 
            if  info_feat== 'Moving':
                info_feat=temp_info[1]+ temp_info[2]
                gen=temp_info[3]
            params= temp_info[-1].replace('.csv','')
            data_temp['generation']= np.repeat(gen, len(data_temp))
      
            data_temp['info_feat']= np.repeat(info_feat, len(data_temp))

            params=params.replace('[','')
            params=params.replace(']','')
            params=params.replace('\'','')
        
            try:
                params_exp=data_temp['explanation'][0].split('\'')[0]
            except: 
                print(f'NO DATA FOUND FOR {file}')
           
            if 'Saliency' in params_exp:
                params_exp=params_exp.replace('_PTY','')
                sp =  params.split(' ')
                if sp[-1]=='False':
                    params_exp =sp[0]
                else: 
                    params_exp = sp[0]+ ' - TSR'
            elif 'LEFTIST' in params_exp:
                sp =  params.split(' ')
                params_exp = params_exp+' - '  +sp[3]
            elif 'TSEvo' in params_exp:
                sp =  params.split(' ')
                params_exp = params_exp+' - '  +sp[0]
            elif 'NativeGuide' in params_exp:
                sp =  params.split(' ')
                params_exp = params_exp+' - '  +sp[1]


            data_temp['params']= np.repeat(params, len(data_temp))
            data_temp['params_exp']= np.repeat(params_exp, len(data_temp))
            if methods is not None:
                data_temp = data_temp[data_temp['params_exp'].isin(methods)]
           

            array=[]
            perturb_array=['FO','FO - TSR','LEFTIST - Shap','LEFTIST - Lime' ]
            gradient_array=['GRAD','SG','GRAD - TSR','SG - TSR','GS','GS - TSR']
            example_array=['TSEvo - information','NativeGuide - dtw','NativeGuideCF - NG','NativeGuideCF - dtw','NativeGuideCF - NUN_CF']
            for _,item in data_temp.iterrows():
                if item['params_exp'] in perturb_array :
                    array.append('Perturbation')
                if item['params_exp'] in gradient_array :
                    array.append('Gradient')
                if item['params_exp'] in example_array :
                    array.append('Example')
            
            data_temp['Based']= array
           

            if '_1_' in file: 
                acc=pd.read_csv('./Benchmarking/ClassificationModels/models_new/preformance_univariate.csv',index_col=0)
            else: 
                acc=pd.read_csv('./Benchmarking/ClassificationModels/models_new/preformance_multivariate.csv',index_col=0)

            new_df = pd.merge(data_temp, acc,  how='left', on=['name','model'],validate='m:1')

            data_temp=new_df

            flag = True

            if  len(data_temp['model'])==0:
                continue
            if acc_model_threshold is not None: 
               
                temp= acc[acc['model']==data_temp['model'][0]]
                f=file.split('.npy')[0]+'.npy'
                f=f.replace('Testing','Training')
                temp = temp[temp['name']==f]

                try: 
                    flag=temp['acc'].values[0]> acc_model_threshold
                except: 
                    t=temp['model']

                    print(f'MODEL DATA NOT FOUND FOR {t} {f} ')

            if data_full is None and flag: 
                data_full = data_temp
            elif flag: 
                data_full=pd.concat([data_full, data_temp],ignore_index=True)
        if strin=='':
            for a in np.sort(np.unique(data_full['params_exp'])):
                strin+= '& ' + f'{a}'
            strin += '\\\\'
        if 'Co' in metric:
                m=['complexity']
                #strin+= 'Complexity '
        if 'Fa' in metric:
                m=['faithfulness_correlation_synthetic_flex']
                #strin+= 'Faithfulness Correlation'
        if 'Ro' in metric:
                m=['AverageSensitivity','MaxSensitivity']
                #strin+= 'Avergage Robustness'
        if 'Re' in metric:
                m=['Relevance Rank','Relevance Mass']
                #strin+= 'Avergage Robustness'
        for b in m: 
            strin+= f' {b} '
            for a in np.sort(np.unique(data_full['params_exp'])):
                mean= round(np.mean(data_full[data_full['params_exp']==a][f'{b}']),2)
                std= round(np.std(data_full[data_full['params_exp']==a][f'{b}']),2)
                strin+=  f'& ${mean} \pm {std} $'
            strin+= ' \\\\ '
    return strin



    
'''PLOTS FOR ELEMENTWISE SECTION'''
def plot_itemwise_ORG_vs_GT(data_name,data_dir,model, explainer,i,resultpath,save_path,roboust=False):
        '''
        Plots Explanation vs Ground Truth. 

        Attributes: 
            data_name str: dataset name as in the file name XXXXX.npy
            data_dir str: directory to data 
            model str: Model Type to be used e.g., CNN
            explainer Callable: Instance of explainer to be used 
            i int: Item to be shown in Plot
        '''
        # Load Data
        #font = {'family' : 'normal',
        #'weight' : 'bold',
        #'size'   : 22}

        #matplotlib.rc('font', **font)
        #plt.rcParams.update({'font.size': 22})
        plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
        #plt.rcParams["text.usetex"] =True
        data_train, meta_train, label_train, data_full, meta_full, label_full=load_synthetic_data(data_name,data_dir,True)
        data_name=data_name.replace('Training','Testing')
        data=data_full[data_name]
        label_full=label_full[data_name]
        meta_full=meta_full[data_name]



        data_shape_1=data.shape[1]
        data_shape_2=data.shape[2]

        
        data_full,_, scaler= scaling( data, label_full, data_shape_1, data_shape_2)
        # Load Model 
        modelName =  data_name.replace('Testing','Training')
        mod= torch.load(f'./Benchmarking/ClassificationModels/models_new/{model}/{modelName}',map_location='cpu')
        # Load & Manipulate Explainer 
        explainer= manipulate_exp_method(data_full, label_full, data_shape_1, data_shape_2, scaler, explainer, mod,check_consist=False)

        if explainer.mode =='feat':
            referencesSamples=np.zeros((data_shape_2,data_shape_1))
        else: 
            referencesSamples=np.zeros((data_shape_1,data_shape_2))
        #print(f'Reference Samples Shapes {referencesSamples.shape}')
        TargetTS_Starts=int(meta_full[i,1])
        TargetTS_Ends=int(meta_full[i,2])
        TargetFeat_Starts= int(meta_full[i,3])
        TargetFeat_Ends= int(meta_full[i,4])

        if explainer.mode =='feat':
            #print(referencesSamples[int(TargetFeat_Starts):int(TargetFeat_Ends),int(TargetTS_Starts):int(TargetTS_Ends)])
            referencesSamples[int(TargetFeat_Starts):int(TargetFeat_Ends),int(TargetTS_Starts):int(TargetTS_Ends)]=1
        else: 
            referencesSamples[int(TargetTS_Starts):int(TargetTS_Ends),int(TargetFeat_Starts):int(TargetFeat_Ends)]=1
        GT=referencesSamples

        

        #Run Explainer 
        d=data[i]
        #TODO 
        d2=d
        d2=d+ np.ones_like(d)*0.5
        l=label_full[i]
        sal=get_explanation( d[np.newaxis,...], l[np.newaxis,...], data_shape_1, data_shape_2, explainer, mod)
        sal2=get_explanation( d2[np.newaxis,...], l[np.newaxis,...], data_shape_1, data_shape_2, explainer, mod)
        #print(sal)
        sal=np.array(sal).reshape(-1, d.shape[-2],d.shape[-1])
        exp=np.absolute(sal[0])

        sal2=np.array(sal2).reshape(-1, d.shape[-2],d.shape[-1])
        exp2=np.absolute(sal2[0])
        
        fig, axs = plt.subplots(2, 1,sharex=True)
        if roboust:
            fig, axs = plt.subplots(3, 1,sharex=True)
        if explainer.mode == "feat":
            d = d.reshape(d.shape[-1], d.shape[-2])
            exp = np.array(exp).reshape(d.shape[-1], d.shape[-2])
            exp2 = np.array(exp2).reshape(d.shape[-1], d.shape[-2])
        else:
            d = d.reshape(d.shape[-2], d.shape[-1])
            exp = np.array(exp).reshape(d.shape[-2], d.shape[-1])
            exp2 = np.array(exp2).reshape(d.shape[-2], d.shape[-1])

        if 'counterfactual' in str(type(explainer)):
            plt.style.use("classic")
            colors = [
            "#08F7FE",  # teal/cyan
            "#FE53BB",  # pink
            "#F5D300",  # yellow
            "#00ff41",  # matrix green
            ]
            df = pd.DataFrame(
                {
                f"Original": list(d.flatten()),
                f"Counterfactual:": list(exp.flatten()),
                }
            )

            df.plot(marker=".", color=colors, ax=axs[0])
            # Redraw the data with low alpha and slighty increased linewidth:
            n_shades = 10
            diff_linewidth = 1.05
            alpha_value = 0.3 / n_shades
            for n in range(1, n_shades + 1):
                df.plot(
                marker=".",
                linewidth=2 + (diff_linewidth * n),
                alpha=alpha_value,
                legend=False,
                ax=axs[0],
                color=colors,
                yticklabels=False,
                )

            axs[0].grid(color="#2A3459")
        else: 
            axn012 = axs[0].twinx()
            axn012.set(xticklabels=[])
            axn012.set(yticklabels=[])
            axs[0].set(yticklabels=[])
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=axs[0],
                vmin=0,
                vmax=1,
            )
            sns.lineplot(
                x=range(0, len(d.reshape(-1))),
                y=d.flatten(),
                ax=axn012,
                color="white",
            )
            axs[0].set(ylabel='E(x)')
            axs[0].set(xticklabels=[])
            axs[0].set(yticklabels=[])
        
            if roboust:
                axn012 = axs[1].twinx()
                axs[1].set(yticklabels=[])
                axn012.set(xticklabels=[])
                axn012.set(yticklabels=[])
                sns.heatmap(
                exp2.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=axs[1],
                yticklabels=False,
                vmin=0,
                vmax=1,
                )
                sns.lineplot(
                x=range(0, len(d2.reshape(-1))),
                y=d2.flatten(),
                ax=axn012,
                color="white",
                )
                axs[1].set(ylabel='E(x+e)')
                axs[1].set(xticklabels=[])
                axs[1].set(yticklabels=[])
        sns.heatmap(
                GT.reshape(1, -1),
                fmt="g",
                cbar=False,
                cmap="viridis",
                ax=axs[-1],
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
        axs[-1].set(ylabel='GT')
        axs[-1].set(xticklabels=[])
        axs[-1].set(yticklabels=[])
       
        #axs[1].imshow(np.array(GT))
        plt.legend([],[], frameon=False)
        #try:
        #    evalu = pd.read_csv(f'{resultpath}/{data_name}_{model}_{str(parameters_to_pandas(explainer).values)}_Plain.csv')
        #    acc=evalu['Acc'].values[0]
        #    plt.title(f'ACC {acc}')
        #except: 
        #    print('No File Found')
        #plt.axis('off')
        plt.savefig(f'{save_path}.png')
        plt.savefig(f'{save_path}.svg')
