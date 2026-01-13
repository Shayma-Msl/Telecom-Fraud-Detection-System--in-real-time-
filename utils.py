import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as spp
import math
import pandas as pd
from collections import Counter
from sklearn.utils import check_random_state, check_array



"""
	Utility functions to handle early stopping and mixed droupout and mixed liner.
"""

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices.astype(np.int64)),  # Tensor indices must be long,
            torch.FloatTensor(coo.data),
            coo.shape)



def matrix_to_torch(X):
    if spp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

def misclassification_cost( y_true, y_pred,cost_table):
    """Appends misclassification costs to model predictions.
    Parameters
    ----------
    y_true : array-like of shape = [n_samples, 1]
             True class values.

    y_pred : array-like of shape = [n_samples, 1]
             Predicted class values.
    """
    df = pd.DataFrame({'row': y_pred, 'column': y_true})
    df = df.merge(cost_table, how='left', on=['row', 'column'])

    return df['cost'].values

# cost matrix
SET_COST_MATRIX_HOW = ('uniform', 'inverse', 'log1p-inverse')
def _set_cost_matrix(y,how: str = 'inverse'):
    """Set the cost matrix according to the 'how' parameter."""
    classes_, _y_encoded = np.unique(y, return_inverse=True)
    _encode_map = {c: np.where(classes_ == c)[0][0] for c in classes_}
    origin_distr_ = dict(Counter(_y_encoded))
    classes, origin_distr = _encode_map.values(), origin_distr_
    cost_matrix = []
    for c_pred in classes:
        cost_c = [
            origin_distr[c_pred] / origin_distr[c_actual]
            for c_actual in classes
        ]
        cost_c[c_pred] = 1
        cost_matrix.append(cost_c)
    if how == 'uniform':
        return np.ones_like(cost_matrix)
    elif how == 'inverse':
        return cost_matrix
    elif how == 'log1p-inverse':
        return np.log1p(cost_matrix)
    else:
        raise ValueError(
            f"When 'cost_matrix' is string, it should be"
            f" in {SET_COST_MATRIX_HOW}, got {how}."
        )

def cost_table_calc( cost_matrix):
    """Creates a table of values from the cost matrix.
    Write the matrix form cost matrix in the form of coordinates + cost value

    Parameters
    ----------
    cost_matrix : array-like of shape = [n_classes, n_classes]

    Returns
    -------
    df : dataframe of shape = [n_classes * n_classes, 3]

    """
    table = np.empty((0, 3))

    for (x, y), value in np.ndenumerate(cost_matrix):
        # table = np.vstack((table, np.array([x + 1, y + 1, value])))
        table = np.vstack((table, np.array([x , y , value])))

    return pd.DataFrame(table, columns=['row', 'column', 'cost'])

def _validate_cost_matrix(cost_matrix, n_classes):
    """validate the cost matrix."""
    cost_matrix = check_array(cost_matrix,
        ensure_2d=True, allow_nd=False,
        force_all_finite=True)
    if cost_matrix.shape != (n_classes, n_classes):
        raise ValueError(
            "When 'cost_matrix' is array-like, it should"
            " be of shape = [n_classes, n_classes],"
            " got shape = {0}".format(cost_matrix.shape)
        )
    return cost_matrix

def realtime_fraud_detection(new_call_csv, graph, gat_model, lgb_model, preprocess_call_logs_func, save_updates=True):
    import os
    import joblib
    from dgl import add_self_loop
    from dgl.data.utils import save_graphs

    SCALER_PATH = r"scaler.joblib"
    FEATURES_CSV = r"all_feat_with_label.csv"
    UPDATED_FEATURES_CSV = r"all_feat_with_label_updated.csv"
    UPDATED_GRAPH_PATH = r"graph_updated.bin"

    LGB_FEATURES = [
        'CallHour', 'CallMinute', 'CallSecond', 'CallingNumber', 'CalledNumber',
        'Callduration', 'callDay', 'callMonth', 'callYear',
        'IntrunkTT', 'OuttrunkTT', 'InSwitch_IGW_TUN', 'OutSwitch_IGW_TUN',
        'Intrunk_enc', 'Outtrunk_enc'
    ]

    # === Clean and standardize input ===
    new_call = new_call_csv.copy()
    new_call.columns = new_call.columns.str.strip().str.replace(" ", "_")
    new_call['CallingNumber'] = new_call['CallingNumber'].astype(str)
    new_call['CalledNumber'] = new_call['CalledNumber'].astype(str)
    for col in ['CallHour', 'CallMinute', 'CallSecond', 'Callduration', 'callDay', 'callMonth', 'callYear']:
        new_call[col] = pd.to_numeric(new_call[col], errors='coerce')
    caller = new_call['CallingNumber'].iloc[0]
    called = new_call['CalledNumber'].iloc[0]

    # === Update features ===
    existing_features = pd.read_csv(FEATURES_CSV) if os.path.exists(FEATURES_CSV) else None
    df_logs = pd.read_csv(r"filtered_calls.csv")
    df_logs = pd.concat([df_logs, new_call], ignore_index=True)
    df_logs.to_csv(r"filtered_calls.csv", index=False)

    df_node_features = preprocess_call_logs_func(df_logs, include_label=True)
    feature_cols = [col for col in df_node_features.columns if col not in ['phone_no_m', 'label']]
    features_raw = df_node_features[feature_cols].values.astype(np.float32)

    # === Standardize
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Missing scaler.joblib")
    scaler = joblib.load(SCALER_PATH)
    features = scaler.transform(features_raw)

    # === Update graph with nodes
    phone_to_node = {}
    for idx, phone in enumerate(df_node_features['phone_no_m']):
        if idx < graph.num_nodes():
            phone_to_node[phone] = idx
        else:
            graph.add_nodes(1, data={'feat': torch.zeros((1, 19))})
            phone_to_node[phone] = graph.num_nodes() - 1

    # === Assign features
    for idx, phone in enumerate(df_node_features['phone_no_m']):
        node_id = phone_to_node[phone]
        graph.ndata['feat'][node_id] = torch.tensor(features[idx]).float()

    # === Add edges for new call
    if caller in phone_to_node and called in phone_to_node:
        graph.add_edges(phone_to_node[caller], phone_to_node[called])
        graph.add_edges(phone_to_node[called], phone_to_node[caller])
    graph = add_self_loop(graph)

    # === Predict with GAT-COBO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    gat_model.g = graph
    gat_model = gat_model.to(device)
    gat_model.eval()

    with torch.no_grad():
        logits, _, _ = gat_model(graph.ndata['feat'].float())
        caller_node = phone_to_node[caller]
        node_logits = logits[caller_node]
        probs = torch.softmax(node_logits, dim=0)
        prediction = torch.argmax(node_logits)

    if prediction.item() == 1:
        raw_for_lgb = new_call[LGB_FEATURES].copy()
        for col in raw_for_lgb.columns:
            raw_for_lgb[col] = pd.to_numeric(raw_for_lgb[col], errors='coerce')
        raw_for_lgb = raw_for_lgb.fillna(0)
        fraud_label = int(lgb_model.predict(raw_for_lgb)[0] > 0.5)
        fraud_type = {0: "PBX_Hacking_Fraud", 1: "Wangiri_Fraud"}.get(fraud_label, "Unknown")

        if save_updates:
            save_graphs(UPDATED_GRAPH_PATH, [graph])
            df_node_features.to_csv(UPDATED_FEATURES_CSV, index=False)

        return {
            "isFraud": True,
            "fraud_type": fraud_type,
            "node_id": caller_node,
            "caller": caller
        }

    return {
        "isFraud": False,
        "node_id": phone_to_node[caller],
        "caller": caller
    }


def preprocess_call_logs_func(df_calls, include_label=True):
    import pandas as pd

    print("📦 Preprocessing call logs...")

    if df_calls is None or df_calls.empty:
        print("⚠️ Received empty or None DataFrame.")
        return pd.DataFrame()

    try:
        df = df_calls.copy()
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        df['CallingNumber'] = df['CallingNumber'].astype(str)
        df['CalledNumber'] = df['CalledNumber'].astype(str)

        if include_label and 'isFraud' in df.columns:
            df['isFraud'] = pd.to_numeric(df['isFraud'], errors='coerce').fillna(0).astype(int)

        numeric_columns = ['Callduration', 'CallHour', 'CallMinute', 'CallSecond',
                           'callDay', 'callMonth', 'callYear',
                           'IntrunkTT', 'OuttrunkTT',
                           'InSwitch_IGW_TUN', 'OutSwitch_IGW_TUN',
                           'Intrunk_enc', 'Outtrunk_enc']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        features = pd.DataFrame({'phone_no_m': df['CallingNumber'].unique()})
        features.set_index('phone_no_m', inplace=True)
        grouped = df.groupby('CallingNumber')

        features['opposite_count'] = grouped['CalledNumber'].count()
        features['opposite_unique'] = grouped['CalledNumber'].nunique()

        def contact_freq_stats(series):
            vc = series.value_counts()
            return pd.Series({
                'phone2opposite_mean': vc.mean() if not vc.empty else 0,
                'phone2opposite_median': vc.median() if not vc.empty else 0,
                'phone2opposite_max': vc.max() if not vc.empty else 0
            })

        contact_stats = grouped['CalledNumber'].apply(contact_freq_stats).unstack()
        features = features.join(contact_stats)

        def duration_per_contact(group):
            summed = group.groupby('CalledNumber')['Callduration'].sum()
            return pd.Series({
                'phone2oppo_sum_mean': summed.mean() if not summed.empty else 0,
                'phone2oppo_sum_median': summed.median() if not summed.empty else 0,
                'phone2oppo_sum_max': summed.max() if not summed.empty else 0
            })

        dur_stats = grouped.apply(duration_per_contact)
        features = features.join(dur_stats)

        features['call_dur_mean'] = grouped['Callduration'].mean()
        features['call_dur_median'] = grouped['Callduration'].median()
        features['call_dur_max'] = grouped['Callduration'].max()
        features['call_dur_min'] = grouped['Callduration'].min()

        features['voc_hour_mode'] = grouped['CallHour'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else -1
        )
        features['voc_hour_mode_count'] = grouped['CallHour'].agg(
            lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0
        )
        features['voc_hour_nunique'] = grouped['CallHour'].nunique()

        features['voc_day_mode'] = grouped['callDay'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else -1
        )
        features['voc_day_mode_count'] = grouped['callDay'].agg(
            lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0
        )
        features['voc_day_nunique'] = grouped['callDay'].nunique()

        features['busi_count'] = grouped['CallHour'].apply(
            lambda x: ((x >= 8) & (x <= 17)).sum()
        )

        if include_label and 'isFraud' in df.columns:
            features['label'] = grouped['isFraud'].max()

        features.fillna(0, inplace=True)
        features.reset_index(inplace=True)  # ✅ moved here so phone_no_m becomes a column

        print("✅ Feature generation complete.")
        return features

    except Exception as e:
        print("❌ Error during feature generation:", e)
        return pd.DataFrame()
    


