import torch
import numpy as np
import pandas as pd
import transformers
from sklearn.metrics import accuracy_score, roc_auc_score

class NNConfig(transformers.PretrainedConfig):
    pass

class NNModel(transformers.PreTrainedModel):
    config_class=NNConfig
    def __init__(self,config):
        super().__init__(config)
        self.input_size = config.input_size
        self.dropout_rate = config.dropout_rate
        self.class_weights = config.class_weights
        self.hidden_1 =torch.nn.Linear(in_features=self.input_size,out_features=config.hidden_size)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.hidden_2 = torch.nn.Linear(in_features=config.hidden_size,out_features=config.hidden_size_2)
        self.output = torch.nn.Linear(in_features=config.hidden_size_2,out_features=2)
        self.relu = torch.nn.ReLU()
        if self.class_weights is not None:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.Tensor(self.class_weights))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self,features,labels=None,input_ids=None,attention_mask=None):
        x = self.relu(self.hidden_1(features))
        x = self.dropout(x)
        x = self.relu(self.hidden_2(x))
        logits = self.output(x)
        if labels is not None:
            return (self.loss_fct(logits,labels),logits)
        else:
            return (logits,)


def class_metrics_nn(logits, labels):
    """
    Arguments:
         logits (numpy.ndarray): logits
         labels (numpy.ndarray): labels
    Returns:
        metrics (dict): AUC and accuracy
        """
    #apply softmax on predictions which are of shape (num_samples,1)
    softmax = torch.nn.Softmax(dim=1)
    pred_proba = softmax(torch.tensor(logits,dtype=torch.float32))
    y_pred = pred_proba.argmax(1)
    y_true = labels
    roc_auc = roc_auc_score(y_true=y_true, y_score=pred_proba[:, 1])
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {"roc_auc": roc_auc, "acc": accuracy}
    return metrics

def compute_metrics(p: transformers.EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = class_metrics_nn(preds, p.label_ids)
    return result

class CancerDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_path,device):
        self.df = pd.read_csv(dataset_path,header=0,index_col=0)
        self.column_names = list(self.df.columns.values)
        self.feature_names = self.column_names[:-2]
        self.features = self.df[self.feature_names].to_numpy()
        self.labels = self.df.label.values
        self.device = device
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_data = {'features':torch.from_numpy(np.asarray(self.features[idx])).to(torch.float).to(self.device),
                    'labels':torch.from_numpy(np.asarray(self.labels[idx])).to(torch.long).to(self.device)
                    }
        return input_data

    def __len__(self):
        return len(self.df)