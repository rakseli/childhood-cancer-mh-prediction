import numpy as np
import pandas as pd
import os
import shutil
import argparse as arg
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from nested_cv_helpers import Parameters
from file_paths import results_path,data_path

#read bash arguments
parser = arg.ArgumentParser()
parser.add_argument("-repeat",type=int)
parser.add_argument("-outer_fold",type=int)
parser.add_argument("-inner_fold",type=int,default=0)
parser.add_argument("-param_num",type=int,default=0)
parser.add_argument("-model",type=str)
parser.add_argument("-fold_type",type=str)
parser.add_argument("-dataset",type=str,default='single_diagnose')
parser.add_argument("-seed",type=int)
args = parser.parse_args()

param_collection = Parameters()
output_path = f"{results_path}/nested_cv_{args.model}_results_repeat_{args.repeat}_{args.dataset}"

if args.model == 'lr' or args.model == 'rf':
    dataset_path = f"{data_path}/nested_cv_repeat_{args.repeat}_{args.dataset}/"
    if args.fold_type=='inner':
        train_df = pd.read_csv(f"{dataset_path}outer_{args.outer_fold}_inner_cv_{args.inner_fold}_train.csv",header=0,index_col=0)
        test_df = pd.read_csv(f"{dataset_path}outer_{args.outer_fold}_inner_cv_{args.inner_fold}_test.csv",header=0,index_col=0)
        column_names = list(train_df.columns.values)
        feature_names = column_names[:-2]
        assert 'label' not in feature_names
        train_f = train_df[feature_names].to_numpy()
        test_f = test_df[feature_names].to_numpy()
        train_l = train_df.label.values
        test_l = test_df.label.values
    else:
        #outer fold
        train_df = pd.read_csv(f"{dataset_path}outer_cv_{args.outer_fold}_train.csv",header=0,index_col=0)
        test_df = pd.read_csv(f"{dataset_path}outer_cv_{args.outer_fold}_test.csv",header=0,index_col=0)
        column_names = list(train_df.columns.values)
        feature_names = column_names[:-2]
        assert 'label' not in feature_names
        train_f = train_df[feature_names].to_numpy()
        test_f = test_df[feature_names].to_numpy()
        train_l = train_df.label.values
        test_l = test_df.label.values

if args.model == 'lr':
    from sklearn.linear_model import LogisticRegression
    try:
        os.mkdir(output_path)
    except FileExistsError as error:
        print(error)
    if args.fold_type=='inner':
        params = param_collection.lr_params[str(args.param_num)]
    else:
        #count mean from every parameter file
        means = []
        for num in range(1,37):
            inner_df = pd.read_csv(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{num}.csv",header=None)
            means.append(np.mean(inner_df.iloc[:,0]))
        best_params_idx=str(means.index(max(means))+1)
        params = param_collection.lr_params[best_params_idx]
    logreg = LogisticRegression(penalty='elasticnet',class_weight='balanced',C=params[0],solver='saga',max_iter=500,l1_ratio=params[1],random_state=args.seed)
    logreg.fit(train_f,train_l)
    pred_proba=logreg.predict_proba(test_f)
    y_pred = logreg.predict(test_f)
    auc = roc_auc_score(test_l,pred_proba[:,1])
    acc = accuracy_score(test_l,y_pred)
    if args.fold_type == 'inner':
        mode = 'a' if os.path.exists(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{args.param_num}.csv") else 'w'
        with open(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{args.param_num}.csv", mode) as f:
            f.write(f"{auc},{acc}\n")
    else:
        mode = 'a' if os.path.exists(f"{output_path}/aucs_outer_cv.csv") else 'w'
        with open(output_path+"/aucs_outer_cv.csv", mode) as f:
            f.write(f"{auc},{acc},{params}\n")

elif args.model == 'nn':
    import transformers
    import torch
    from nn_model import NNConfig,NNModel,CancerDataset,compute_metrics

    transformers.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = f"{data_path}/nested_cv_repeat_{args.repeat}_{args.dataset}_val_set"
    checkpoint_dir = f"{results_path}/nn_checkpoints"
    try:
        os.mkdir(output_path)
        os.mkdir(checkpoint_dir)
    except FileExistsError as error:
        print(error)
    if args.fold_type=='inner':
        train_data = CancerDataset(dataset_path=f"{dataset_path}/outer_{args.outer_fold}_inner_cv_{args.inner_fold}_train.csv",device=device)
        val_data = CancerDataset(dataset_path=f"{dataset_path}/outer_{args.outer_fold}_inner_cv_{args.inner_fold}_val.csv",device=device)
        test_data = CancerDataset(dataset_path=f"{dataset_path}/outer_{args.outer_fold}_inner_cv_{args.inner_fold}_test.csv",device=device)
        params = param_collection.nn_params[str(args.param_num)]
    else:
        train_data = CancerDataset(dataset_path=f"{dataset_path}/outer_cv_{args.outer_fold}_train.csv",device=device)
        val_data = CancerDataset(dataset_path=f"{dataset_path}/outer_cv_{args.outer_fold}_val.csv",device=device)
        test_data = CancerDataset(dataset_path=f"{dataset_path}/outer_cv_{args.outer_fold}_test.csv",device=device)
        #count mean from every parameter file
        means = []
        for num in range(1,37):
            inner_df = pd.read_csv(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{num}.csv",header=None)
            means.append(np.mean(inner_df.iloc[:,0]))
        best_params_idx=str(means.index(max(means))+1)
        params = param_collection.nn_params[best_params_idx]
    classes = np.array((0, 1))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_data.labels).tolist()
    nn_config=NNConfig(input_size=len(train_data.feature_names),hidden_size=256,hidden_size_2=params[2],dropout_rate=params[3],class_weights=weights)
    nn_model = NNModel(nn_config).to(device)
    data_collator = transformers.DefaultDataCollator()
    early_stopping = transformers.EarlyStoppingCallback(20)
    training_args = transformers.TrainingArguments(
        output_dir=checkpoint_dir,  # output directory 
        per_device_train_batch_size=params[0],  # batch size per device during training
        per_device_eval_batch_size=params[0],  # batch size for evaluation
        weight_decay=0.1,  # strength of weight decay
        learning_rate=params[1],
        load_best_model_at_end=True,
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="roc_auc",  # metric for early stopping
        greater_is_better=True,  # greater metric score is better
        evaluation_strategy="epoch",
        num_train_epochs=500,
        dataloader_pin_memory=False
        )
    trainer = transformers.Trainer(
        model=nn_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )

    trainer.train()
    test_results = trainer.predict(test_data)
    results_dict = test_results.metrics
    auc = results_dict['test_roc_auc']
    acc = results_dict['test_acc']
    if args.fold_type == 'inner':
        mode = 'a' if os.path.exists(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{args.param_num}.csv") else 'w'
        with open(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{args.param_num}.csv", mode) as f:
            f.write(f"{auc},{acc}\n")
    else:
        mode = 'a' if os.path.exists(f"{output_path}/aucs_outer_cv.csv") else 'w'
        with open(output_path+"/aucs_outer_cv.csv", mode) as f:
            f.write(f"{auc},{acc},{params}\n")
    #delete checkpoints dir
    shutil.rmtree(checkpoint_dir)

elif args.model == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    try:
        os.mkdir(output_path)
    except FileExistsError as error:
        print(error)
    if args.fold_type=='inner':
        params = param_collection.rf_params[str(args.param_num)]
    else:
        #count mean from every parameter file
        means = []
        for num in range(1,37):
            inner_df = pd.read_csv(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{num}.csv",header=None)
            means.append(np.mean(inner_df.iloc[:,0]))
        best_params_idx=str(means.index(max(means))+1)
        params = param_collection.rf_params[best_params_idx]

    randomforest = RandomForestClassifier(n_estimators=300,max_depth=params[0],min_samples_split=params[3],min_samples_leaf=params[2],max_features=params[1],class_weight='balanced',random_state=args.seed,n_jobs=20)
    randomforest.fit(train_f,train_l)
    pred_proba=randomforest.predict_proba(test_f)
    y_pred = randomforest.predict(test_f)
    auc = roc_auc_score(test_l,pred_proba[:,1])
    acc = accuracy_score(test_l,y_pred)
    if args.fold_type == 'inner':
        mode = 'a' if os.path.exists(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{args.param_num}.csv") else 'w'
        with open(f"{output_path}/aucs_inner_outer_fold_{args.outer_fold}_param_{args.param_num}.csv", mode) as f:
            f.write(f"{auc},{acc}\n")
    else:
        mode = 'a' if os.path.exists(f"{output_path}/aucs_outer_cv.csv") else 'w'
        with open(output_path+"/aucs_outer_cv.csv", mode) as f:
            f.write(f"{auc},{acc},{params}\n")
else:
    raise NotImplementedError(f"Nested cv is implemented only for Logistic regression (lr), RandomForest (rf) and Neural Network (nn), given {args.model}")
