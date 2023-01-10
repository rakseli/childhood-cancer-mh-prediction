import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from dataclasses import dataclass,field
from typing import Dict


def create_folds(cancer_df,diabetes_df,seed,add_val_set=False,output_path='/nested_cv'):
    '''
    Create nested cross validation folds where test sets contain only cancer patients
    Arguments:
        data (pandas.DataFrame): df indexed with uniq uids
        add_val_set (bool): whether to create validation set
        output_path (str): dir where data is saved  
    '''

    if add_val_set:
        output_path_single = f'{output_path}_single_val_set'
        output_path_multi = f'{output_path}_multi_val_set'
    else:
        output_path_single = f'{output_path}_single'
        output_path_multi = f'{output_path}_multi'
    try:
       os.mkdir(output_path_single)
       os.mkdir(output_path_multi)
    except FileExistsError:
        pass

    cancer_df = cancer_df.sample(frac=1.0,random_state=seed)
    diabetes_df = diabetes_df.sample(frac=1.0,random_state=seed)
    labels = cancer_df['label'].to_numpy()
    train_val_split = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=seed)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    outer_fold=0
    for train_index, test_index in cv.split(np.zeros(len(cancer_df)),labels):
        outer_fold_train_df = cancer_df.iloc[train_index]
        outer_fold_test_df = cancer_df.iloc[test_index]
        test_set_indexes = outer_fold_test_df.index.values
        outer_fold_test_df_multi = diabetes_df.loc[test_set_indexes]
        outer_fold_train_df_multi = diabetes_df.loc[diabetes_df.index.difference(test_set_indexes)]
        if add_val_set:
            t_labels = outer_fold_train_df['label'].to_numpy()
            for t_id , v_id in train_val_split.split(np.zeros(len(outer_fold_train_df)),t_labels):
                val_id = v_id
                train_id = t_id
                break
            outer_fold_val_df = outer_fold_train_df.iloc[val_id]
            outer_fold_train_df = outer_fold_train_df.iloc[train_id]
            val_set_indexes = outer_fold_val_df.index.values
            outer_fold_val_df_multi = outer_fold_train_df_multi.loc[val_set_indexes]
            outer_fold_train_df_multi = outer_fold_train_df_multi.loc[outer_fold_train_df_multi.index.difference(val_set_indexes)]
    

        if add_val_set:
            outer_fold_val_df.to_csv(f"{output_path_single}/outer_cv_{outer_fold}_val.csv")
            outer_fold_val_df_multi.to_csv(f"{output_path_multi}/outer_cv_{outer_fold}_val.csv")
        outer_fold_train_df.to_csv(f"{output_path_single}/outer_cv_{outer_fold}_train.csv")
        outer_fold_test_df.to_csv(f"{output_path_single}/outer_cv_{outer_fold}_test.csv")
        outer_fold_train_df_multi.to_csv(f"{output_path_multi}/outer_cv_{outer_fold}_train.csv")
        outer_fold_test_df_multi.to_csv(f"{output_path_multi}/outer_cv_{outer_fold}_test.csv")
        #split inner
        outer_labels = outer_fold_train_df['label'].to_numpy()
        print("Starting to spilt inner")
        train_val_split_inner = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=seed)
        cv_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        inner_fold=0
        for train_index_inner,test_index_inner in cv_inner.split(np.zeros(len(outer_fold_train_df)),outer_labels):
            inner_fold_train_df = outer_fold_train_df.iloc[train_index_inner]
            inner_fold_test_df = outer_fold_train_df.iloc[test_index_inner]
            test_set_indexes_inner = inner_fold_test_df.index.values
            inner_fold_test_df_multi = outer_fold_train_df_multi.loc[test_set_indexes_inner]
            inner_fold_train_df_multi = outer_fold_train_df_multi.loc[outer_fold_train_df_multi.index.difference(test_set_indexes_inner)]
            if add_val_set:
                t_labels_inner = inner_fold_train_df['label'].to_numpy()
                for t_id , v_id in train_val_split_inner.split(np.zeros(len(inner_fold_train_df)),t_labels_inner):
                    val_id_inner = v_id
                    train_id_inner = t_id
                    break
                inner_fold_val_df = inner_fold_train_df.iloc[val_id_inner]
                inner_fold_train_df = inner_fold_train_df.iloc[train_id_inner]
                inner_val_set_indexes = inner_fold_val_df.index.values
                inner_fold_val_df_multi = inner_fold_train_df_multi.loc[inner_val_set_indexes]
                inner_fold_train_df_multi = outer_fold_train_df_multi.loc[inner_fold_train_df_multi.index.difference(inner_val_set_indexes)]
            if add_val_set:
                inner_fold_val_df.to_csv(f"{output_path_single}/outer_{outer_fold}_inner_cv_{inner_fold}_val.csv")
                inner_fold_val_df_multi.to_csv(f"{output_path_multi}/outer_{outer_fold}_inner_cv_{inner_fold}_val.csv")


            inner_fold_train_df.to_csv(f"{output_path_single}/outer_{outer_fold}_inner_cv_{inner_fold}_train.csv")
            inner_fold_test_df.to_csv(f"{output_path_single}/outer_{outer_fold}_inner_cv_{inner_fold}_test.csv")
            inner_fold_train_df_multi.to_csv(f"{output_path_multi}/outer_{outer_fold}_inner_cv_{inner_fold}_train.csv")
            inner_fold_test_df_multi.to_csv(f"{output_path_multi}/outer_{outer_fold}_inner_cv_{inner_fold}_test.csv")
            inner_fold+=1
        outer_fold+=1
    return None


@dataclass(frozen=True)
class Parameters:
    lr_params: Dict[str,list] = field(default_factory=lambda: {"1": [0.1, 0.1], "2": [0.1, 0.26], "3": [0.1, 0.42], "4": [0.1, 0.58], "5": [0.1, 0.74], 
    "6": [0.1, 0.9], "7": [0.48, 0.1], "8": [0.48, 0.26], "9": [0.48, 0.42], "10": [0.48, 0.58], 
    "11": [0.48, 0.74], "12": [0.48, 0.9], "13": [0.86, 0.1], "14": [0.86, 0.26], "15": [0.86, 0.42], 
    "16": [0.86, 0.58], "17": [0.86, 0.74], "18": [0.86, 0.9], "19": [1.24, 0.1], "20": [1.24, 0.26], 
    "21": [1.24, 0.42], "22": [1.24, 0.58], "23": [1.24, 0.74], "24": [1.24, 0.9], "25": [1.62, 0.1], 
    "26": [1.62, 0.26], "27": [1.62, 0.42], "28": [1.62, 0.58], "29": [1.62, 0.74], "30": [1.62, 0.9],
    "31": [2.0, 0.1], "32": [2.0, 0.26], "33": [2.0, 0.42], "34": [2.0, 0.58], "35": [2.0, 0.74],
    "36": [2.0, 0.9]
        })

    rf_params:  Dict[str,list] = field(default_factory=lambda: {"1": [15, "sqrt", 1, 2], "2": [15, "sqrt", 1, 3], "3": [15, "sqrt", 2, 2], "4": [15, "sqrt", 2, 3], "5": [15, "sqrt", 3, 2], 
    "6": [15, "sqrt", 3, 3], "7": [15, None, 1, 2], "8": [15, None, 1, 3], "9": [15, None, 2, 2], "10": [15, None, 2, 3], 
    "11": [15, None, 3, 2], "12": [15, None, 3, 3], "13": [30, "sqrt", 1, 2], "14": [30, "sqrt", 1, 3], "15": [30, "sqrt", 2, 2], 
    "16": [30, "sqrt", 2, 3], "17": [30, "sqrt", 3, 2], "18": [30, "sqrt", 3, 3], "19": [30, None, 1, 2], "20": [30, None, 1, 3], 
    "21": [30, None, 2, 2], "22": [30, None, 2, 3], "23": [30, None, 3, 2], "24": [30, None, 3, 3], "25": [45, "sqrt", 1, 2], 
    "26": [45, "sqrt", 1, 3], "27": [45, "sqrt", 2, 2], "28": [45, "sqrt", 2, 3], "29": [45, "sqrt", 3, 2], "30": [45, "sqrt", 3, 3], 
    "31": [45, None, 1, 2], "32": [45, None, 1, 3], "33": [45, None, 2, 2], "34": [45, None, 2, 3], "35": [45, None, 3, 2], "36": [45, None, 3, 3]
        })

    nn_params:  Dict[str,list] = field(default_factory=lambda: {'1': [8, 0.0001, 64, 0.1], '2': [8, 0.001, 64, 0.1], '3': [8, 0.0001, 128, 0.1], '4': [8, 0.001, 128, 0.1], '5': [8, 0.0001, 64, 0.2], 
    '6': [8, 0.001, 64, 0.2], '7': [8, 0.0001, 128, 0.2], '8': [8, 0.001, 128, 0.2], '9': [8, 0.0001, 64, 0.3], '10': [8, 0.001, 64, 0.3], 
    '11': [8, 0.0001, 128, 0.3], '12': [8, 0.001, 128, 0.3], '13': [16, 0.0001, 64, 0.1], '14': [16, 0.001, 64, 0.1], '15': [16, 0.0001, 128, 0.1], 
    '16': [16, 0.001, 128, 0.1], '17': [16, 0.0001, 64, 0.2], '18': [16, 0.001, 64, 0.2], '19': [16, 0.0001, 128, 0.2], '20': [16, 0.001, 128, 0.2], 
    '21': [16, 0.0001, 64, 0.3], '22': [16, 0.001, 64, 0.3], '23': [16, 0.0001, 128, 0.3], '24': [16, 0.001, 128, 0.3], '25': [32, 0.0001, 64, 0.1], 
    '26': [32, 0.001, 64, 0.1], '27': [32, 0.0001, 128, 0.1], '28': [32, 0.001, 128, 0.1], '29': [32, 0.0001, 64, 0.2], '30': [32, 0.001, 64, 0.2], 
    '31': [32, 0.0001, 128, 0.2], '32': [32, 0.001, 128, 0.2], '33': [32, 0.0001, 64, 0.3], '34': [32, 0.001, 64, 0.3], '35': [32, 0.0001, 128, 0.3], '36': [32, 0.001, 128, 0.3]})



