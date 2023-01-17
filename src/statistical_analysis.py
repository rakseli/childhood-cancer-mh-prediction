# %% 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir
from file_paths import results_path

def get_files(parent_dir,pattern=None,not_field=None):
    all_files = []
    all_dirs = listdir(parent_dir)
    dirs = []
    #get all model paths
    for d in all_dirs:
        if os.path.isdir(os.path.join(parent_dir,d)):
            if pattern is None:
                dirs.append(os.path.join(parent_dir,d))
            if isinstance(pattern,list):
                if not_field is not None:
                    if all(p in d for p in pattern) and not_field not in d:
                        dirs.append(os.path.join(parent_dir,d))
                else:
                    if all(p in d for p in pattern):
                        dirs.append(os.path.join(parent_dir,d))
            else:
                if not_field is not None:
                    if pattern in d and not_field not in d:
                        dirs.append(os.path.join(parent_dir,d))
                else:
                    if pattern in d:
                        dirs.append(os.path.join(parent_dir,d))
    #get files
    for d in dirs:
        files = listdir(d)
        all_files.extend(files)
    result_files = []

    for f in all_files:
        if 'aucs_outer_cv.csv' in f:
            result_files.append(f)

    paths = [f"{p}/{f}" for f,p in zip(result_files,dirs)]
    return paths

def create_plots(aucs_dict,save=True):
    width = 0.1
    plt.figure(figsize=(10,7))
    xaxis= np.arange(15)
    #rf
    plt.bar(xaxis,aucs_dict['s_rf'],width,label=(f"Random Forest single (total mean AUC {np.around(np.mean(aucs_dict['s_rf']),3)})"),color='#377eb8')
    plt.bar(xaxis + width,aucs_dict['m_rf'],width,label=(f"Random Forest multi (total mean AUC {np.around(np.mean(aucs_dict['m_rf']),3)})"),color='#ff7f00')
    #lr
    plt.bar(xaxis + width*2,aucs_dict['s_lr'],width,label=(f"Log Reg single (total mean AUC {np.around(np.mean(aucs_dict['s_lr']),3)})"),color='#4daf4a')
    plt.bar(xaxis + width*3,aucs_dict['m_lr'],width,label=(f"Log Reg multi (total mean AUC {np.around(np.mean(aucs_dict['m_lr']),3)})"),color='#f781bf')
    #nn
    plt.bar(xaxis + width*4,aucs_dict['s_nn'],width,label=(f"NN single (total mean AUC {np.around(np.mean(aucs_dict['s_nn']),3)})"),color='#a65628')
    plt.bar(xaxis + width*5,aucs_dict['m_nn'],width,label=(f"NN multi (total mean AUC {np.around(np.mean(aucs_dict['m_nn']),3)})"),color='#dede00')
    plt.ylabel('AUC',fontsize='large')
    plt.title('AUCs of classifiers in three times repeated 5-fold nested cross-validation',fontsize='large')
    plt.xticks(xaxis + width, ('R1 F1','R1 F2','R1 F3','R1 F4','R1 F5','R2 F1','R2 F2','R2 F3','R2 F4','R2 F5','R3 F1','R3 F2','R3 F3','R3 F4','R3 F5'),fontsize='medium')
    plt.legend(loc=4,fontsize='large')
    if save:
        plt.savefig(f'{results_path}/cancer_aucs.pdf', dpi=300)
    plt.show()
     


# %%
#single diagnose
single_diagose_rf_repeat_1 = get_files(parent_dir=results_path,pattern=['_single','rf','repeat_1'])
single_diagose_nn_repeat_1 = get_files(parent_dir=results_path,pattern=['_single','nn','repeat_1'])
single_diagose_lr_repeat_1 = get_files(parent_dir=results_path,pattern=['_single','lr','repeat_1'])

single_diagose_rf_repeat_2 = get_files(parent_dir=results_path,pattern=['_single','rf','repeat_2'])
single_diagose_nn_repeat_2 = get_files(parent_dir=results_path,pattern=['_single','nn','repeat_2'])
single_diagose_lr_repeat_2 = get_files(parent_dir=results_path,pattern=['_single','lr','repeat_2'])

single_diagose_rf_repeat_3 = get_files(parent_dir=results_path,pattern=['_single','rf','repeat_3'])
single_diagose_nn_repeat_3 = get_files(parent_dir=results_path,pattern=['_single','nn','repeat_3'])
single_diagose_lr_repeat_3 = get_files(parent_dir=results_path,pattern=['_single','lr','repeat_3'])

multi_diagnose_paths_rf_repeat_1 = get_files(parent_dir=results_path,pattern=['rf','repeat_1'],not_field='_single')
multi_diagnose_paths_nn_repeat_1 = get_files(parent_dir=results_path,pattern=['nn','repeat_1'],not_field='_single')
multi_diagnose_paths_lr_repeat_1 = get_files(parent_dir=results_path,pattern=['lr','repeat_1'],not_field='_single')

multi_diagnose_paths_rf_repeat_2 = get_files(parent_dir=results_path,pattern=['rf','repeat_2'],not_field='_single')
multi_diagnose_paths_nn_repeat_2 = get_files(parent_dir=results_path,pattern=['nn','repeat_2'],not_field='_single')
multi_diagnose_paths_lr_repeat_2 = get_files(parent_dir=results_path,pattern=['lr','repeat_2'],not_field='_single')

multi_diagnose_paths_rf_repeat_3 = get_files(parent_dir=results_path,pattern=['rf','repeat_3'],not_field='_single')
multi_diagnose_paths_nn_repeat_3 = get_files(parent_dir=results_path,pattern=['nn','repeat_3'],not_field='_single')
multi_diagnose_paths_lr_repeat_3 = get_files(parent_dir=results_path,pattern=['lr','repeat_3'],not_field='_single')

auc_dict = {}

s_rf_df = pd.read_csv(single_diagose_rf_repeat_1[0],header=None)
s_nn_df = pd.read_csv(single_diagose_nn_repeat_1[0],header=None)
s_lr_df = pd.read_csv(single_diagose_lr_repeat_1[0],header=None)
m_rf_df = pd.read_csv(multi_diagnose_paths_rf_repeat_1[0],header=None)
m_nn_df = pd.read_csv(multi_diagnose_paths_nn_repeat_1[0],header=None)
m_lr_df = pd.read_csv(multi_diagnose_paths_lr_repeat_1[0],header=None)
auc_dict['s_rf_r_1']=s_rf_df.iloc[:,0]
auc_dict['m_rf_r_1']=m_rf_df.iloc[:,0]
auc_dict['s_lr_r_1']=s_lr_df.iloc[:,0]
auc_dict['m_lr_r_1']=m_lr_df.iloc[:,0]
auc_dict['s_nn_r_1']=s_nn_df.iloc[:,0]
auc_dict['m_nn_r_1']=m_nn_df.iloc[:,0]
s_rf_df = pd.read_csv(single_diagose_rf_repeat_2[0],header=None)
s_nn_df = pd.read_csv(single_diagose_nn_repeat_2[0],header=None)
s_lr_df = pd.read_csv(single_diagose_lr_repeat_2[0],header=None)
m_rf_df = pd.read_csv(multi_diagnose_paths_rf_repeat_2[0],header=None)
m_nn_df = pd.read_csv(multi_diagnose_paths_nn_repeat_2[0],header=None)
m_lr_df = pd.read_csv(multi_diagnose_paths_lr_repeat_2[0],header=None)
auc_dict['s_rf_r_2']=s_rf_df.iloc[:,0]
auc_dict['m_rf_r_2']=m_rf_df.iloc[:,0]
auc_dict['s_lr_r_2']=s_lr_df.iloc[:,0]
auc_dict['m_lr_r_2']=m_lr_df.iloc[:,0]
auc_dict['s_nn_r_2']=s_nn_df.iloc[:,0]
auc_dict['m_nn_r_2']=m_nn_df.iloc[:,0]
s_rf_df = pd.read_csv(single_diagose_rf_repeat_3[0],header=None)
s_nn_df = pd.read_csv(single_diagose_nn_repeat_3[0],header=None)
s_lr_df = pd.read_csv(single_diagose_lr_repeat_3[0],header=None)
m_rf_df = pd.read_csv(multi_diagnose_paths_rf_repeat_3[0],header=None)
m_nn_df = pd.read_csv(multi_diagnose_paths_nn_repeat_3[0],header=None)
m_lr_df = pd.read_csv(multi_diagnose_paths_lr_repeat_3[0],header=None)
auc_dict['s_rf_r_3']=s_rf_df.iloc[:,0]
auc_dict['m_rf_r_3']=m_rf_df.iloc[:,0]
auc_dict['s_lr_r_3']=s_lr_df.iloc[:,0]
auc_dict['m_lr_r_3']=m_lr_df.iloc[:,0]
auc_dict['s_nn_r_3']=s_nn_df.iloc[:,0]
auc_dict['m_nn_r_3']=m_nn_df.iloc[:,0]
auc_dict['s_rf']=np.concatenate((auc_dict['s_rf_r_1'],auc_dict['s_rf_r_2'],auc_dict['s_rf_r_3']),axis=None)
auc_dict['m_rf']=np.concatenate((auc_dict['m_rf_r_1'],auc_dict['m_rf_r_2'],auc_dict['m_rf_r_3']),axis=None)
auc_dict['s_lr']=np.concatenate((auc_dict['s_lr_r_1'],auc_dict['s_lr_r_2'],auc_dict['s_lr_r_3']),axis=None)
auc_dict['m_lr']=np.concatenate((auc_dict['m_lr_r_1'],auc_dict['m_lr_r_2'],auc_dict['m_lr_r_3']),axis=None)
auc_dict['s_nn']=np.concatenate((auc_dict['s_nn_r_1'],auc_dict['s_nn_r_2'],auc_dict['s_nn_r_3']),axis=None)
auc_dict['m_nn']=np.concatenate((auc_dict['m_nn_r_1'],auc_dict['m_nn_r_2'],auc_dict['m_nn_r_3']),axis=None)

assert len(auc_dict['m_nn'])==15
create_plots(aucs_dict=auc_dict)


#%%
from sklearn.model_selection import ParameterGrid
import baycomp

models = {'a':['m_rf','s_rf','m_lr','s_lr','m_nn','s_nn'],'b':['s_rf','m_rf','s_lr','m_lr','s_nn','m_nn']}
comparisons = list(ParameterGrid(models))
results_dict={}

for c in comparisons:
    if f"{c['a']}_vs_{c['b']}" not in results_dict.keys() and f"{c['b']}_vs_{c['a']}" not in results_dict.keys() and f"{c['a']}_vs_{c['b']}" != f"{c['b']}_vs_{c['a']}":
        results = baycomp.CorrelatedTTest.probs(x=auc_dict[c['a']],y=auc_dict[c['b']],rope=0.01,runs=3)
        results_dict[f"{c['a']}_vs_{c['b']}"]=(results,baycomp.CorrelatedTTest.sample(auc_dict[c['a']],y=auc_dict[c['b']],runs=3))

print(results_dict['m_rf_vs_s_rf'])
d = {'comparison':results_dict.keys(),'a':[0.0]*len(results_dict.keys()),'rope':[0.0]*len(results_dict.keys()),'b':[0.0]*len(results_dict.keys())}
df = pd.DataFrame(d)
df=df.astype(object)
df['posterior_dist']=pd.NA
for i,(k,value) in enumerate(results_dict.items()):
    #comparison
    df.iloc[i,0]=k
    #a
    df.iloc[i,1]=value[0][0]
    #rope
    df.iloc[i,2]=value[0][1]
    #b
    df.iloc[i,3]=value[0][2]
    #posterior_dict
    df.iat[i,4]=value[1].tolist()

df = df.set_index('comparison')
df.to_csv(f'{results_path}/bayesian_analysis.csv')
#%%
df.head()
#%%
#bayesian estimation
import baycomp
    
results = baycomp.CorrelatedTTest.probs(x=auc_dict['m_rf'],y=auc_dict['s_lr'],rope=0.01,runs=3)
print(results)
fig=baycomp.CorrelatedTTest.plot(x=auc_dict['m_rf'],y=auc_dict['s_lr'],rope=0.01,names=('m_rf','s_rf'))
df = pd.read_csv(f'{results_path}/cancer_results_minimal.csv',header=0,index_col=0)
plt.axvline(df.loc['m_rf_vs_s_lr','CI_low'],label='CI',color='red')
plt.axvline(df.loc['m_rf_vs_s_lr','CI_high'],color='red')
plt.xlabel('Difference')
plt.title("Multi diagnose Random Forest vs Single diagnose Log Reg")
plt.legend()
plt.show()
sample = baycomp.CorrelatedTTest.sample(x=auc_dict['m_rf'],y=auc_dict['s_rf'])