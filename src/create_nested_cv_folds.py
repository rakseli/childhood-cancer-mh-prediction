import pandas as pd
from nested_cv_helpers import create_folds
from file_paths import data_path

cancer_df_single = pd.read_csv(f'{data_path}/cancer_data_lemmatized_True_vectorized_True_diagnosis_C_late_effect_365_devel_mode_False.csv',header=0,index_col=0)
cancer_df_multi = pd.read_csv(f'{data_path}/cancer_data_lemmatized_True_vectorized_True_diagnosis_C-E_late_effect_365_devel_mode_False.csv',header=0,index_col=0)


create_folds(cancer_df_single,cancer_df_multi,seed=8,add_val_set=True,output_path=f'{data_path}/nested_cv_repeat_1')
create_folds(cancer_df_single,cancer_df_multi,seed=8,add_val_set=False,output_path=f'{data_path}/nested_cv_repeat_1')


cancer_df_single = pd.read_csv(f'{data_path}/cancer_data_lemmatized_True_vectorized_True_diagnosis_C_late_effect_365_devel_mode_False.csv',header=0,index_col=0)
cancer_df_multi = pd.read_csv(f'{data_path}/cancer_data_lemmatized_True_vectorized_True_diagnosis_C-E_late_effect_365_devel_mode_False.csv',header=0,index_col=0)

create_folds(cancer_df_single,cancer_df_multi,seed=9,add_val_set=True,output_path=f'{data_path}/nested_cv_repeat_2')
create_folds(cancer_df_single,cancer_df_multi,seed=9,add_val_set=False,output_path=f'{data_path}/nested_cv_repeat_2')

cancer_df_single = pd.read_csv(f'{data_path}/cancer_data_lemmatized_True_vectorized_True_diagnosis_C_late_effect_365_devel_mode_False.csv',header=0,index_col=0)
cancer_df_multi = pd.read_csv(f'{data_path}/cancer_data_lemmatized_True_vectorized_True_diagnosis_C-E_late_effect_365_devel_mode_False.csv',header=0,index_col=0)

create_folds(cancer_df_single,cancer_df_multi,seed=77,add_val_set=True,output_path=f'{data_path}/nested_cv_repeat_3')
create_folds(cancer_df_single,cancer_df_multi,seed=77,add_val_set=False,output_path=f'{data_path}/nested_cv_repeat_3')
