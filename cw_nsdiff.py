for seed_idx in range(10):
    import math
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from tqdm import tqdm
    from dataclasses import dataclass
    import os
    import argparse
    from typing import Dict, Optional, Tuple, List
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from scipy.stats import norm
    from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
    from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
    from torch_timeseries.nn.embedding import DataEmbedding
    import wandb
    import yaml
    import math
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from tqdm import tqdm
    from dataclasses import dataclass
    import os
    import argparse
    from typing import Dict, Optional, Tuple, List
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from scipy.stats import norm
    from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
    from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
    from torch_timeseries.nn.embedding import DataEmbedding
    import wandb

    from torchmetrics import Metric
    import CRPS.CRPS as pscore  # Assuming `pscore` is the function to compute CRPS
    from concurrent.futures import ProcessPoolExecutor

    import torch
    import torch.nn as nn
    from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
    from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
    from torch_timeseries.nn.embedding import DataEmbedding

    from dataclasses import dataclass, field
    import sys
    from typing import List, Dict
    import os
    import torch
    from dataclasses import dataclass, asdict, field
    from torch_timeseries.nn.embedding import freq_map
    import argparse
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
    from torch.optim import *
    from tqdm import tqdm
    from torch_timeseries.utils.model_stats import count_parameters
    from torch_timeseries.utils.reproduce import reproducible
    import time
    # import multiprocessing
    import torch.multiprocessing as mp
    from torch_timeseries.utils.parse_type import parse_type

    from torch_timeseries.utils.early_stop import EarlyStopping
    import yaml
    import numpy as np
    import torch.distributed as dist
    import torch
    from tqdm import tqdm
    import concurrent.futures
    from types import SimpleNamespace

    from dataclasses import asdict, dataclass
    import datetime
    import hashlib
    import json
    import os
    import random
    import time
    from typing import Dict, List, Type, Union

    import numpy as np
    import pandas as pd
    import torch
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
    from tqdm import tqdm
    from torch.nn import MSELoss, L1Loss
    from torch.optim import *
    from torch_timeseries.dataset import *
    from torch_timeseries.scaler import *

    from torch_timeseries.utils.model_stats import count_parameters
    from torch_timeseries.utils.early_stop import EarlyStopping
    from torch_timeseries.utils.parse_type import parse_type
    from torch_timeseries.utils.reproduce import reproducible
    from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
    from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
    from torch_timeseries.experiments import ForecastExp
    from torch_timeseries.utils import asdict_exc
    import torch.multiprocessing as mp

    import matplotlib.pyplot as plt
    import torch

    from torch_timeseries.utils.parse_type import parse_type
    from torch_timeseries.dataloader import ETTHLoader

    import torch
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor


    import math
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from tqdm import tqdm
    from dataclasses import dataclass
    import os
    import argparse
    from typing import Dict, Optional, Tuple, List
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from scipy.stats import norm
    from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
    from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
    from torch_timeseries.nn.embedding import DataEmbedding, TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding
    import wandb

    from torchmetrics import Metric
    import CRPS.CRPS as pscore  # Assuming `pscore` is the function to compute CRPS
    from concurrent.futures import ProcessPoolExecutor

    import torch
    import torch.nn as nn
    from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
    from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
    from torch_timeseries.nn.embedding import DataEmbedding

    from dataclasses import dataclass, field
    import sys
    from typing import List, Dict
    import os
    import torch
    from dataclasses import dataclass, asdict, field
    from torch_timeseries.nn.embedding import freq_map
    import argparse
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
    from torch.optim import *
    from tqdm import tqdm
    from torch_timeseries.utils.model_stats import count_parameters
    from torch_timeseries.utils.reproduce import reproducible
    import time
    # import multiprocessing
    import torch.multiprocessing as mp
    from torch_timeseries.utils.parse_type import parse_type

    from torch_timeseries.utils.early_stop import EarlyStopping
    import yaml
    import numpy as np
    import torch.distributed as dist
    import torch
    from tqdm import tqdm
    import concurrent.futures
    from types import SimpleNamespace

    from dataclasses import asdict, dataclass
    import datetime
    import hashlib
    import json
    import os
    import random
    import time
    from typing import Dict, List, Type, Union

    import numpy as np
    import pandas as pd
    import torch
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
    from tqdm import tqdm
    from torch.nn import MSELoss, L1Loss
    from torch.optim import *
    from torch_timeseries.dataset import *
    from torch_timeseries.scaler import *

    from torch_timeseries.utils.model_stats import count_parameters
    from torch_timeseries.utils.early_stop import EarlyStopping
    from torch_timeseries.utils.parse_type import parse_type
    from torch_timeseries.utils.reproduce import reproducible
    from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
    from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
    from torch_timeseries.experiments import ForecastExp
    from torch_timeseries.utils import asdict_exc
    import torch.multiprocessing as mp

    import matplotlib.pyplot as plt
    import torch

    from torch_timeseries.utils.parse_type import parse_type
    from torch_timeseries.dataloader import ETTHLoader
    from torchmetrics import MetricCollection
    import torch
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor

    # if you want original model without conditionally whitening, please use if_cw = False
    # if you want conditionally whitened model, please use if_cw = True
    if_cw = True

    device = "cuda:3"
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_seed(114 + seed_idx)


    import os
    import pandas as pd

    data_path = "ts_datasets/ETTh1/ETTh1.csv"
    assert os.path.exists(data_path), f"cannot find: {data_path}"

    df = pd.read_csv(data_path)

    # display(df.head())

    config = {
        # training strategy, we never used them. please always set it as False
        "teacher_force": False,
        "mixup": False,

        # the weight of JMCE loss
        "matrix_norm_weight": [(7*192)**0.5 * 0.1, 0.1, 0.], # [f norm, svd norm, fft norm of matrix] the fft norm is always 0
        "fft_weight": [1, 0.], # [L2 norm, fft norm of mean] the fft norm is always 0
        "eign_penalty": 50, # w_Eigen
        "eps_eign_min": 0.1, # lambda_min
        "penalty_method": "hard", # hard: use relu, soft: use softplus
        'num_training_steps': 20, # epoch of training JMCE

        # NS-Transformer of JMCE
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 3,
        "dropout": 0.1,

        # projector 
        "p_hidden_layers": 2,
        "p_hidden_dims": [128, 128],

        # the length of time series
        "windows": 168,
        "horizon": 1,
        "pred_len": 192,
        "label_len": 168 // 2,
        'num_features': 7,

        # DataLoader 
        "batch_size": 64,
        "num_worker": 0,
        'dataset_type': "ETTh1",
        'data_path': "ts_datasets",
        'scaler_type': "StandardScaler",

        'lr': 1e-4,
        'weight_decay': 5e-4,

        # the length & padding model of sliding window to compute covariance matrix
        'window_size': 95,
        'pad_mode': 'reflect',
    }
    lambda_min = config['eps_eign_min']
    weigen = config['eign_penalty']
    window_length = config['window_size']
    log_file = f"cw_nsdiff_lambda_{lambda_min}_weigen{weigen}.txt"

    # parameters
    d_model = config['d_model']
    n_heads = config['n_heads']
    e_layers = config['e_layers']
    d_layers = config['d_layers']
    d_ff = config['d_ff']
    factor = config['factor']
    dropout = config['dropout']
    p_hidden_layers = config['p_hidden_layers']
    p_hidden_dims=config['p_hidden_dims']
    num_features = config['num_features']



    dataset_type = config['dataset_type']
    data_path = config['data_path']

    DatasetClass = parse_type(dataset_type, globals())
    dataset = DatasetClass(root=data_path)


    scaler_type = config['scaler_type']
    ScalerClass = parse_type(scaler_type, globals())
    scaler = ScalerClass()


    windows = config['windows']
    horizon = config['horizon']
    pred_len = config['pred_len']
    batch_size = config['batch_size']
    num_worker = config['num_worker']
    label_len= windows // 2


    dataloader = ETTHLoader(
        dataset,
        scaler,
        window=windows,
        horizon=horizon,
        steps=pred_len,
        shuffle_train=True,
        freq=dataset.freq,
        batch_size=batch_size,
        num_worker=num_worker,
    )


    # non stationary transformer
    class Projector(nn.Module):
        '''
        MLP to learn the De-stationary factors
        '''
        def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
            super(Projector, self).__init__()

            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                        padding_mode='circular', bias=False)

            layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
            for i in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

            layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
            self.backbone = nn.Sequential(*layers)


        def forward(self, x, stats):
            batch_size = x.shape[0]
            x = self.series_conv(x)  # B x 1 x D
            x = torch.cat([x, stats], dim=1)  # B x 2 x D
            x = x.view(batch_size, -1)  # B x 2D
            y = self.backbone(x)  # B x output_dim

            return y


    class ns_Transformer(nn.Module):
        """
        Non-stationary Transformer
        """
        def __init__(self, 
                    pred_len=pred_len,
                    seq_len=windows,
                    label_len=label_len,
                    output_attention=False,
                    num_features = num_features,
                    enc_in=num_features + int(num_features*(num_features+1)/2),
                    d_model=d_model,
                    embed='timeF',
                    freq=dataloader.dataset.freq,
                    dropout=dropout,
                    dec_in=num_features + int(num_features*(num_features+1)/2),
                    factor=factor,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    e_layers=e_layers,
                    d_layers=d_layers,
                    c_out=num_features + int(num_features*(num_features+1)/2),
                    p_hidden_dims=p_hidden_dims,
                    p_hidden_layers=p_hidden_layers,
                    activation = nn.SiLU(),
                    kernel_size = 3,
                    ):
            super(ns_Transformer, self).__init__()
            self.pred_len = pred_len 
            self.seq_len = seq_len 
            self.label_len = label_len 
            self.output_attention = output_attention 
            self.num_feature = num_features
            self.num_feature_triangle = int(num_features*(num_features+1)/2)

            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                            dropout) 
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                            dropout) 

            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            DSAttention(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention), d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )

            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(True, factor, attention_dropout=dropout, output_attention=False),
                            d_model, n_heads),
                        AttentionLayer(
                            DSAttention(False, factor, attention_dropout=dropout, output_attention=False),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model),

                projection=nn.Linear(d_model, c_out, bias=True)
            )


            self.tau_learner = Projector(enc_in=enc_in, seq_len=seq_len, hidden_dims=p_hidden_dims,
                                        hidden_layers=p_hidden_layers, output_dim=1, kernel_size = kernel_size)
            self.delta_learner = Projector(enc_in=enc_in, seq_len=seq_len,
                                        hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers,
                                        output_dim=seq_len, kernel_size = kernel_size)
            self.future_mixup_layer = nn.Linear(self.pred_len,self.seq_len)

        def unpack_cholesky_upper(self, flat_triu): # in paper, we write lower triangle matrix. but before we write paper, we first use upper tri. they are equivalent so i didnt change it
            B, T, _ = flat_triu.shape
            D = self.num_feature
            U = torch.zeros(B, T, D, D, device=flat_triu.device)
            triu_idx = torch.triu_indices(D, D, device=flat_triu.device)
            U[:, :, triu_idx[0], triu_idx[1]] = flat_triu
            U[:, :, range(D), range(D)] = F.softplus(U[:, :, range(D), range(D)])  
            return U

        def forward(self, x_enc, x_enc_xxT_trig, x_mark_enc, x_dec, x_mark_dec,
                    enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                    future_mixup_weight = 1, batch_y=None, batch_yyT_trig=None):
            """
            x_enc (Tensor): Encoded input sequence of shape (B, seq_len, enc_in)
            x_mark_enc (Tensor): Encoded input time features of shape (B, seq_len, d)
            x_dec (Tensor): Decoded input sequence of shape (B, seq_len, dec_in)
            x_mark_dec (Tensor): Decoded input time features of shape (B, seq_len, dec_in)
            """
            x_enc = torch.cat([x_enc,x_enc_xxT_trig],dim=-1)
            # print(x_enc.shape)
            
            if (batch_y is not None) and (batch_yyT_trig is not None):
                batch_y = torch.cat([batch_y,batch_yyT_trig],dim=-1)
                x_enc = future_mixup_weight * x_enc + (1 - future_mixup_weight) * self.future_mixup_layer(batch_y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x_enc = x_enc
            x_raw = x_enc.clone().detach()

            # Normalization
            mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
            x_enc = x_enc - mean_enc
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
            x_enc = x_enc / std_enc

            x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                                dim=1).to(x_enc.device).clone()

            tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
            delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

            enc_out = self.enc_embedding(x_enc, x_mark_enc)

            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

            dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

            # De-normalization
            # this is important, only denormalize the mean part!!!
            dec_out[:,:,:num_features] = dec_out[:,:,:num_features] * std_enc[:,:,:num_features] + mean_enc[:,:,:num_features]

            
            miu_pred = dec_out[:,:,:num_features]
            U_flat = dec_out[:,:,num_features:]

            
            U = self.unpack_cholesky_upper(U_flat)   
            Sigma_pred = U.transpose(-1, -2) @ U

            if self.output_attention:
                return miu_pred[:, -self.pred_len:, :], Sigma_pred[:, -self.pred_len:, :, :], attns
            else:

                return miu_pred[:, -self.pred_len:, :], Sigma_pred[:, -self.pred_len:, :, :], dec_out  # [B, L, D]





    def sliding_cov(batch_x, window_size, pad_mode='reflect'):
        """
        Args:
            batch_x: (B, T, D)
            window_size: int
            pad_mode: 'reflect' or 'replicate'
        Returns:
            moving_cov: (B, T, D, D)
            moving_cov_tri: (B, T, D * (D + 1) // 2)
        """
        B, T, D = batch_x.shape
        pad_left = window_size // 2
        pad_right = window_size - 1 - pad_left

        # 1. Pad
        x = batch_x.permute(0, 2, 1)  # (B, D, T)
        x_padded = F.pad(x, (pad_left, pad_right), mode=pad_mode)  # (B, D, T + pad)

        # 2. Build Conv1d kernel
        avg_kernel = torch.ones(D, 1, window_size, device=batch_x.device) / window_size

        # 3. Sliding mean: (B, D, T)
        mean_x = F.conv1d(x_padded, avg_kernel, groups=D)  # depthwise conv
        mean_x = mean_x.permute(0, 2, 1)  # (B, T, D)

        # 4. Compute XX^T for each t: (B, T, D, D)
        xxT = batch_x.unsqueeze(3) @ batch_x.unsqueeze(2)  # (B, T, D, D)
        xxT_flat = xxT.reshape(B, T, D * D).permute(0, 2, 1)  # (B, D*D, T)

        # 5. Pad and Conv1d for mean of XX^T
        xxT_padded = F.pad(xxT_flat, (pad_left, pad_right), mode=pad_mode)
        avg_kernel_xxT = torch.ones(D * D, 1, window_size, device=batch_x.device) / window_size
        mean_xxT = F.conv1d(xxT_padded, avg_kernel_xxT, groups=D * D)  # (B, D*D, T)
        mean_xxT = mean_xxT.permute(0, 2, 1).reshape(B, T, D, D)  # (B, T, D, D)

        # 6. Covariance: E[XX^T] - E[X]E[X]^T
        mean_x_outer = mean_x.unsqueeze(3) @ mean_x.unsqueeze(2)  # (B, T, D, D)
        cov = mean_xxT - mean_x_outer  # (B, T, D, D)

        # 7. Extract upper triangular part
        idx = torch.triu_indices(D, D, offset=0, device=batch_x.device)
        cov_tri = cov[:, :, idx[0], idx[1]]  # (B, T, D*(D+1)/2)

        return cov, cov_tri


    def cacf_torch(x, max_lag, dim=(0, 1)):
        def get_lower_triangular_indices(n):
            return [list(x) for x in torch.tril_indices(n, n)]

        ind = get_lower_triangular_indices(x.shape[2])
        x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
        x_l = x[..., ind[0]]
        x_r = x[..., ind[1]]
        cacf_list = list()
        for i in range(max_lag):
            y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
            cacf_i = torch.mean(y, (1))
            cacf_list.append(cacf_i)
        cacf = torch.cat(cacf_list, 1)
        return cacf.reshape(cacf.shape[0], -1, len(ind[0]))

    def lower_triangular_to_full_matrix(values, dim):
        """
        values: (B, 1, num_pairs)
        dim: D
        return: (B, D, D)
        """
        B = values.shape[0]
        num_pairs = values.shape[2]
        idx = torch.tril_indices(row=dim, col=dim, offset=0, device=values.device)

        mat = torch.zeros(B, dim, dim, device=values.device)

        mat[:, idx[0], idx[1]] = values.squeeze(1)

        mat = mat + mat.transpose(1,2) - torch.diag_embed(torch.diagonal(mat, dim1=1, dim2=2))
        return mat


    def upper_triangular_to_full_matrix(values, dim):

        B = values.shape[0]
        num_pairs = values.shape[2]
        idx = torch.triu_indices(row=dim, col=dim, offset=0, device=values.device)

        mat = torch.zeros(B, dim, dim, device=values.device)

        mat[:, idx[0], idx[1]] = values.squeeze(1)

        mat = mat + mat.transpose(1, 2) - torch.diag_embed(torch.diagonal(mat, dim1=1, dim2=2))

        return mat

    def compute_corr_score(batch_y,batch_y_cw):
        batch_y_corr_score = lower_triangular_to_full_matrix(cacf_torch(x=batch_y,max_lag=1),dim = num_features)
        batch_y_cw_corr_score = lower_triangular_to_full_matrix(cacf_torch(x=batch_y_cw,max_lag=1),dim = num_features)
        return batch_y_corr_score, batch_y_cw_corr_score


    def average_r2_correlation_metric(X, normalize = True, method = 'linear'):
        """
        X: Tensor of shape (B, T, D)
        Returns:
            avg_r2: float
            r2_scores: list of R² per variable
        """
        B, T, D = X.shape
        # reshape to (B*T, D)
        data = X.reshape(B*T, D).cpu().numpy()

        # standardize each variable
        if normalize:
            data_mean = data.mean(axis=0, keepdims=True)
            data_std = data.std(axis=0, keepdims=True) + 1e-8
            data = (data - data_mean) / data_std

        r2_scores = []
        if method == 'linear':
            for target_idx in range(D):
                X_other = np.delete(data, target_idx, axis=1)  # shape (N, D-1)
                y_target = data[:, target_idx]                # shape (N,)

                model = LinearRegression().fit(X_other, y_target)
                y_pred = model.predict(X_other)
                r2 = r2_score(y_target, y_pred)
                r2_scores.append(r2)
        elif method == 'xgb':
            for target_idx in range(D):
                X_other = np.delete(data, target_idx, axis=1)  # (N, D-1)
                y_target = data[:, target_idx]                 # (N,)

                model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, verbosity=0)
                model.fit(X_other, y_target)
                y_pred = model.predict(X_other)
                r2 = r2_score(y_target, y_pred)
                r2_scores.append(r2)

        avg_r2 = np.mean(r2_scores)
        return avg_r2, r2_scores


    def soft_hinge_all_mean(eigvals, eps=1e-3, beta=5.0):

        eps_t = torch.as_tensor(eps, dtype=eigvals.dtype, device=eigvals.device)
        gaps  = eps_t - eigvals                 # [B,T,D]
        softg = F.softplus(beta * gaps) / beta  
        return softg.mean()                     

    # the loss function of JMCE
    def joint_loss_fn(theta_true, 
                    theta_outer_true, 
                    miu_pred, 
                    sigma_pred, 
                    batch_xxT_std,
                    matrix_norm_weight = [1/3, 1/3, 1/3], 
                    fft_weight= [1/2, 1/2],
                    eign_penalty=0.1,
                    eps_eign_min = 1e-3,
                    penalty_method = 'hard',
                    verbose = False,):
        """
        theta_true:       [B, T, D]
        theta_outer_true: [B, T, D, D] (theta * theta^T)
        miu_pred:         [B, T, D]
        sigma_pred:       [B, T, D, D]
        """
        # Mean prediction error
        loss_miu = F.mse_loss(miu_pred, theta_true)
        fft_loss_miu = (torch.fft.rfft(miu_pred, dim=1) - torch.fft.rfft(theta_true, dim=1)).abs().mean()

        ######################### dont forget to normalize the sigma here! #######################
        diff = (sigma_pred - theta_outer_true) / batch_xxT_std
        loss_fro = diff.pow(2).mean()
        svals = torch.linalg.svdvals(diff)  # [B, T, D]
        loss_svd = svals.mean()
        

        # Positive-definiteness constraint: penalize cases where the minimum eigenvalue of Σ - μμᵀ is below 0
        if eign_penalty > 0:
            cov_consistency = sigma_pred
            eigvals = torch.linalg.eigvalsh(cov_consistency)
            if penalty_method == 'hard':
                posdef_penalty = torch.relu(eps_eign_min - eigvals).mean()
            elif penalty_method == 'soft':
                posdef_penalty = soft_hinge_all_mean(eigvals, eps=eps_eign_min, beta=20.0)
        else:
            posdef_penalty = 0
        
        if verbose:
            if eign_penalty > 0:
                print(f'l2 loss:{loss_miu.item()}, f norm loss:{loss_fro.item()}, svd norm loss:{loss_svd.item()}, penalty:{posdef_penalty.item()}')
            else:
                print(f'l2 loss:{loss_miu.item()}, f norm loss:{loss_fro.item()}, svd norm loss:{loss_svd.item()}, penalty: not used')
        fft_loss_cov = (torch.fft.rfft(diff, dim=1) ).abs().mean()
        return (fft_weight[0] * loss_miu + fft_weight[1] * fft_loss_miu) \
            + (loss_fro * matrix_norm_weight[0] + loss_svd * matrix_norm_weight[1] +fft_loss_cov * matrix_norm_weight[2]) \
            + eign_penalty * posdef_penalty






    def compute_neural_cov(output_theta_theta_T,output_conditional_mean):
        
        return output_theta_theta_T.detach()


    def whiten_sequence(theta, conditional_mean, cov_matrix, eps=1e-3, verbose=False):
        """
        theta: [B, T, D]
        conditional_mean: [B, T, D]
        cov_matrix: [B, T, D, D]

        Returns:
            whitened: [B, T, D]
            inv_sqrt_all: [B, T, D, D]
            sqrt_all: [B, T, D, D]
        """
        B, T, D = theta.shape
        residual = theta - conditional_mean  # [B, T, D]
        whitened = torch.zeros_like(residual)
        inv_sqrt_all = torch.zeros(B, T, D, D, device=theta.device)
        sqrt_all = torch.zeros(B, T, D, D, device=theta.device)

        for b in range(B):
            for t in range(T):
                cov = cov_matrix[b, t]  # [D, D]

                eigvals, eigvecs = torch.linalg.eigh(cov)

                if verbose and (eigvals < eps).any():
                    print(f"eigvals before clamp: {eigvals}")

                eigvals_clamped = torch.clamp(eigvals, min=eps)

                # inverse sqrt
                inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals_clamped)) @ eigvecs.T
                inv_sqrt_all[b, t] = inv_sqrt
                whitened[b, t] = inv_sqrt @ residual[b, t]

                # sqrt
                sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                sqrt_all[b, t] = sqrt

        return whitened, inv_sqrt_all, sqrt_all





    set_seed(514 + seed_idx)
    model_conditional_mean = ns_Transformer().float().to(device)
    print('training JMCE, the parameter num is:')
    print(sum(p.numel() for p in model_conditional_mean.parameters()))
    optimizer = torch.optim.AdamW(
        model_conditional_mean.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'] 
    )


    val_loss = []
    num_training_steps = config['num_training_steps']
    loss_func = nn.MSELoss()
    teacher_force = config['teacher_force']
    mixup = config['mixup']
    matrix_norm_weight = config['matrix_norm_weight']
    fft_weight= config['fft_weight']
    eign_penalty = config['eign_penalty']
    eps_eign_min = config['eps_eign_min']
    penalty_method = config['penalty_method']
    window_size = config['window_size']
    pad_mode = config['pad_mode']


    val_loss = []
    num_training_steps = config['num_training_steps']
    loss_func = nn.MSELoss()
    teacher_force = config['teacher_force']
    mixup = config['mixup']
    matrix_norm_weight = config['matrix_norm_weight']
    fft_weight= config['fft_weight']
    eign_penalty = config['eign_penalty']
    eps_eign_min = config['eps_eign_min']
    penalty_method = config['penalty_method']
    window_size = config['window_size']
    pad_mode = config['pad_mode']

    best_val_loss = float('inf')
    best_model_state = None
    best_step = -1

    for step in range(num_training_steps):
        model_conditional_mean.train()
        total_loss = 0
    ###################################################   train   #######################################
        for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader.train_loader):


            optimizer.zero_grad()


            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            batch_x_sliding_cov, batch_x_sliding_cov_trig= sliding_cov(batch_x=batch_x, 
                                                                    window_size=window_size, 
                                                                    pad_mode=pad_mode)
            batch_y_sliding_cov, batch_y_sliding_cov_trig= sliding_cov(batch_x=batch_y, 
                                                                    window_size=window_size, 
                                                                    pad_mode=pad_mode)

            batch_xxT, batch_xxT_trig = batch_x_sliding_cov.detach(), batch_x_sliding_cov_trig.detach()
            batch_yyT, batch_yyT_trig = batch_y_sliding_cov.detach(), batch_y_sliding_cov_trig.detach()
            
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()

            # ############# normalize the sigma, this is to keep the scale of cov-loss ###########
            # batch_xxT_trig = batch_xxT_trig / train_set_xxT_up_trig_std
            # batch_yyT_trig = batch_yyT_trig / train_set_xxT_up_trig_std
            # batch_xxT = batch_xxT / train_set_xxT_std
            # batch_yyT = batch_yyT / train_set_xxT_std
            # ############# normalize the sigma ############################

            # === mixup, but we never use it ===
            if mixup:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                idx = torch.randperm(batch_x.size(0))
                batch_x = lam * batch_x + (1 - lam) * batch_x[idx]
                batch_y = lam * batch_y + (1 - lam) * batch_y[idx]
                batch_xxT, batch_xxT_trig = lam * batch_xxT + (1 - lam) * batch_xxT[idx], lam * batch_xxT_trig + (1 - lam) * batch_xxT_trig[idx]
                batch_yyT, batch_yyT_trig = lam * batch_yyT + (1 - lam) * batch_yyT[idx], lam * batch_yyT_trig + (1 - lam) * batch_yyT_trig[idx]
                batch_x_mark = lam * batch_x_mark + (1 - lam) * batch_x_mark[idx]
                batch_y_mark = lam * batch_y_mark + (1 - lam) * batch_y_mark[idx]
            # === mixup ===

            batch_y_input = torch.concat([batch_x[:, -label_len:, :], batch_y], dim=1)
            batch_yyT_input = torch.cat([batch_xxT_trig[:, -label_len:, :], batch_yyT_trig], dim=1)
            batch_y_mark_input = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1)

            
            dec_inp_label = torch.cat([batch_x[:, -label_len :, :].to(device),batch_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
            dec_inp_pred = torch.zeros(
                                        [batch_x.size(0), pred_len, 
                                        dataset.num_features + int(dataset.num_features*(dataset.num_features+1)/2)]
                                    ).to(device)
            dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

            if teacher_force: # never use teacher_force
                future_mixup_weight = torch.rand((batch_y.shape[0],1,1)).to(device)
                miu_pred, sigma_pred, _ = model_conditional_mean(x_enc = batch_x, 
                                                        x_enc_xxT_trig = batch_xxT_trig, 
                                                        x_mark_enc = batch_x_mark, 
                                                        x_dec = dec_inp, 
                                                        x_mark_dec = batch_y_mark_input,
                                                        enc_self_mask=None, 
                                                        dec_self_mask=None, 
                                                        dec_enc_mask=None,
                                                        future_mixup_weight = future_mixup_weight, 
                                                        batch_y=batch_y, 
                                                        batch_yyT_trig=batch_yyT_trig)
            else:
                future_mixup_weight = torch.rand((batch_y.shape[0],1,1)).to(device)
                miu_pred, sigma_pred, _ = model_conditional_mean(x_enc = batch_x, 
                                                        x_enc_xxT_trig = batch_xxT_trig, 
                                                        x_mark_enc = batch_x_mark, 
                                                        x_dec = dec_inp, 
                                                        x_mark_dec = batch_y_mark_input,
                                                        enc_self_mask=None, 
                                                        dec_self_mask=None, 
                                                        dec_enc_mask=None,
                                                        future_mixup_weight = 1, 
                                                        batch_y=None, 
                                                        batch_yyT_trig=None)


            loss = joint_loss_fn(theta_true=batch_y, 
                                theta_outer_true=batch_yyT, 
                                miu_pred=miu_pred, 
                                sigma_pred=sigma_pred,
                                #  batch_xxT_std = train_set_xxT_std,
                                batch_xxT_std = 1,
                                matrix_norm_weight = matrix_norm_weight, 
                                fft_weight= fft_weight,
                                eign_penalty=eign_penalty, 
                                eps_eign_min = eps_eign_min,
                                penalty_method = penalty_method,
                                verbose = False,)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model_conditional_mean.parameters(), 1.)
            optimizer.step()


            total_loss += loss.item()
        total_loss = total_loss / len(dataloader.train_loader)
        

    ###################################################   val   #######################################
        with torch.no_grad():
            model_conditional_mean.eval()
            val_total = 0
            for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader.val_loader):
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                batch_x_sliding_cov, batch_x_sliding_cov_trig= sliding_cov(batch_x=batch_x, 
                                                                        window_size=window_size, 
                                                                        pad_mode=pad_mode)
                batch_y_sliding_cov, batch_y_sliding_cov_trig= sliding_cov(batch_x=batch_y, 
                                                                        window_size=window_size, 
                                                                        pad_mode=pad_mode)
            
                batch_xxT, batch_xxT_trig = batch_x_sliding_cov.detach(), batch_x_sliding_cov_trig.detach()
                batch_yyT, batch_yyT_trig = batch_y_sliding_cov.detach(), batch_y_sliding_cov_trig.detach()
                
                batch_x_mark = batch_x_mark.to(device).float()
                batch_y_mark = batch_y_mark.to(device).float()

                # ############# normalize the sigma ############################
                # batch_xxT_trig = batch_xxT_trig / train_set_xxT_up_trig_std
                # batch_yyT_trig = batch_yyT_trig / train_set_xxT_up_trig_std
                # batch_xxT = batch_xxT / train_set_xxT_std
                # batch_yyT = batch_yyT / train_set_xxT_std
                # ############# normalize the sigma ############################


                batch_y_input = torch.concat([batch_x[:, -label_len:, :], batch_y], dim=1)
                batch_yyT_input = torch.cat([batch_xxT_trig[:, -label_len:, :], batch_yyT_trig], dim=1)
                batch_y_mark_input = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1)

                dec_inp_pred = torch.zeros(
                    [batch_x.size(0), pred_len, dataset.num_features + int(dataset.num_features*(dataset.num_features+1)/2)]
                ).to(device)
                dec_inp_label = torch.cat([batch_x[:, -label_len :, :].to(device),batch_xxT_trig[:, -label_len:, :].to(device)],dim=-1)

                dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
                
                miu_pred, sigma_pred, _ = model_conditional_mean(x_enc = batch_x, 
                                                        x_enc_xxT_trig = batch_xxT_trig, 
                                                        x_mark_enc = batch_x_mark, 
                                                        x_dec = dec_inp, 
                                                        x_mark_dec = batch_y_mark_input,
                                                        enc_self_mask=None, 
                                                        dec_self_mask=None, 
                                                        dec_enc_mask=None,
                                                        future_mixup_weight = 1, 
                                                        batch_y=None, 
                                                        batch_yyT_trig=None)
                    

                loss = joint_loss_fn(theta_true=batch_y, 
                                    theta_outer_true=batch_yyT, 
                                    miu_pred=miu_pred, 
                                    sigma_pred=sigma_pred,
                                    # batch_xxT_std = train_set_xxT_std,
                                    batch_xxT_std = 1,
                                    matrix_norm_weight = matrix_norm_weight, 
                                    fft_weight= fft_weight,
                                    eign_penalty=eign_penalty, 
                                    eps_eign_min = eps_eign_min,
                                    penalty_method = penalty_method,
                                    verbose = False,)
                val_total += loss.item()
            val_avg = val_total / len(dataloader.val_loader)
            val_loss.append(val_avg)

    ###################################################   test   #######################################
            # visualize the last batch
            test_total = 0
            test_total_mean = 0
            test_total_sigma = 0
            for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader.test_loader):
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                
                batch_x_sliding_cov, batch_x_sliding_cov_trig= sliding_cov(batch_x=batch_x, 
                                                                        window_size=window_size, 
                                                                        pad_mode=pad_mode)
                batch_y_sliding_cov, batch_y_sliding_cov_trig= sliding_cov(batch_x=batch_y, 
                                                                        window_size=window_size, 
                                                                        pad_mode=pad_mode)
            
                batch_xxT, batch_xxT_trig = batch_x_sliding_cov.detach(), batch_x_sliding_cov_trig.detach()
                batch_yyT, batch_yyT_trig = batch_y_sliding_cov.detach(), batch_y_sliding_cov_trig.detach()
                
                batch_x_mark = batch_x_mark.to(device).float()
                batch_y_mark = batch_y_mark.to(device).float()

                # ############# normalize the sigma ############################
                # batch_xxT_trig = batch_xxT_trig / train_set_xxT_up_trig_std
                # batch_yyT_trig = batch_yyT_trig / train_set_xxT_up_trig_std
                # batch_xxT = batch_xxT / train_set_xxT_std
                # batch_yyT = batch_yyT / train_set_xxT_std
                # ############# normalize the sigma ############################


                batch_y_input = torch.concat([batch_x[:, -label_len:, :], batch_y], dim=1)
                batch_yyT_input = torch.cat([batch_xxT_trig[:, -label_len:, :], batch_yyT_trig], dim=1)
                batch_y_mark_input = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1)

                dec_inp_pred = torch.zeros(
                    [batch_x.size(0), pred_len, dataset.num_features + int(dataset.num_features*(dataset.num_features+1)/2)]
                ).to(device)
                dec_inp_label = torch.cat([batch_x[:, -label_len :, :].to(device),batch_xxT_trig[:, -label_len:, :].to(device)],dim=-1)

                dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
                
                miu_pred, sigma_pred, _ = model_conditional_mean(x_enc = batch_x, 
                                                        x_enc_xxT_trig = batch_xxT_trig, 
                                                        x_mark_enc = batch_x_mark, 
                                                        x_dec = dec_inp, 
                                                        x_mark_dec = batch_y_mark_input,
                                                        enc_self_mask=None, 
                                                        dec_self_mask=None, 
                                                        dec_enc_mask=None,
                                                        future_mixup_weight = 1, 
                                                        batch_y=None, 
                                                        batch_yyT_trig=None)
                    

                test_loss = joint_loss_fn(theta_true=batch_y, 
                                    theta_outer_true=batch_yyT, 
                                    miu_pred=miu_pred, 
                                    sigma_pred=sigma_pred,
                                    # batch_xxT_std = train_set_xxT_std,
                                    batch_xxT_std = 1,
                                    matrix_norm_weight = matrix_norm_weight, 
                                    fft_weight= fft_weight,
                                    eign_penalty=eign_penalty, 
                                    eps_eign_min = eps_eign_min,
                                    penalty_method = penalty_method,
                                    verbose = False,)
                
                # #################### de normalize, then we use the predicted cov, whiten the sequence, visualize ####
                # sigma_pred = sigma_pred * train_set_xxT_std
                # batch_yyT = batch_yyT * train_set_xxT_std
                # batch_yyT_trig = batch_yyT_trig * train_set_xxT_up_trig_std
                # #################### de normalize ####################

                neural_cov_last_batch = compute_neural_cov(sigma_pred,miu_pred)

                y_test_cw, cov_sqrt_inv_test, cov_sqrt_test = whiten_sequence(batch_y, 
                                    conditional_mean=miu_pred.detach(), 
                                    cov_matrix=neural_cov_last_batch.detach(), eps=eps_eign_min, verbose = False)
                
                # batch_y_corr_score, batch_y_cw_corr_score = compute_corr_score(batch_y,y_test_cw)

                whiten_score_y, _ = average_r2_correlation_metric(batch_y, normalize = True, method='linear')
                whiten_score_y_cw, _ = average_r2_correlation_metric(y_test_cw, normalize = True, method='linear')
                whiten_score_y_centralized, _ = average_r2_correlation_metric(batch_y-miu_pred, normalize = True, method='linear')

                
                test_total += test_loss.item()
                test_total_mean += loss_func(batch_y, miu_pred).item()
                test_total_sigma += loss_func(batch_yyT, sigma_pred).item()
            test_avg = test_total / len(dataloader.test_loader)
            test_mean_avg = test_total_mean / len(dataloader.test_loader)
            test_sigma_avg = test_total_sigma / len(dataloader.test_loader)

            triu_indices = torch.triu_indices(num_features, num_features)


            # save the best model
            if val_avg < best_val_loss and step >= 10:
                best_val_loss = val_avg
                best_step = step
                best_model_state = {k: v.clone() for k, v in model_conditional_mean.state_dict().items()}

        # print(f"{step}:Train={total_loss:.4f}|Val={val_avg:.4f}|Test={test_avg:.4f}|whiten_score_y={whiten_score_y:.4f}|whiten_score_y_cw={whiten_score_y_cw:.4f}|whiten_score_y_cent={whiten_score_y_centralized:.4f}")

    # ========== load the best model ==========
    if best_model_state is not None:
        model_conditional_mean.load_state_dict(best_model_state)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")





    dataset_cw = ETTh1(root='ts_datasets') # the dataset for CW
    scaler_type = config['scaler_type']
    ScalerClass = parse_type(scaler_type, globals())
    scaler = ScalerClass()

    windows = config['windows']
    horizon = config['horizon']
    pred_len = config['pred_len']
    # batch_size = config['batch_size']
    num_worker = config['num_worker']
    label_len= windows // 2


    dataloader_cw = ETTHLoader(
        dataset_cw,
        scaler,
        window=windows,
        horizon=horizon,
        steps=pred_len,
        shuffle_train=True,
        freq=dataset_cw.freq,
        batch_size=32,
        num_worker=num_worker,
    )

    def whiten_sequence_fast(theta, conditional_mean, cov_matrix, eps=1e-3, verbose=False):
        """
        theta: [B, T, D]
        conditional_mean: [B, T, D]
        cov_matrix: [B, T, D, D]

        Returns:
            whitened: [B, T, D]
            inv_sqrt_all: [B, T, D, D]
            sqrt_all: [B, T, D, D]
        """
        residual = theta - conditional_mean  # [B, T, D]

        # --- batched eigendecomposition ---
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)   # eigvals: [B,T,D], eigvecs: [B,T,D,D]

        if verbose and (eigvals < eps).any():
            print("Some eigenvalues before clamp:", eigvals[eigvals < eps])

        eigvals_clamped = eigvals.clamp_min(eps)  # [B,T,D]

        # --- sqrt and inv sqrt of eigenvalues ---
        sqrt_vals = eigvals_clamped.sqrt()
        inv_sqrt_vals = eigvals_clamped.rsqrt()   # reciprocal sqrt

        # [B,T,D,D]
        sqrt_diag = torch.diag_embed(sqrt_vals)        # [B,T,D,D]
        inv_sqrt_diag = torch.diag_embed(inv_sqrt_vals)

        # --- (vecs @ diag @ vecs^T) ---
        sqrt_all = eigvecs @ sqrt_diag @ eigvecs.transpose(-2, -1)       # [B,T,D,D]
        inv_sqrt_all = eigvecs @ inv_sqrt_diag @ eigvecs.transpose(-2, -1)

        # --- whiten residual ---
        whitened = torch.einsum("btij,btj->bti", inv_sqrt_all, residual)  # [B,T,D]

        return whitened, inv_sqrt_all, sqrt_all

    print('conditionally whitening the dataset')
    future_cw_train = {}
    # with torch.inference_mode():
    model_conditional_mean.eval()
    for i, (batch_history,
            batch_future,
            origin_history,
            origin_future,
            batch_history_mark,
            batch_future_mark,
            ) in enumerate(dataloader_cw.train_loader):
        # print(f'{i}-th batch is whitened')
        batch_history=batch_history.to(device).float()
        batch_future=batch_future.to(device).float()
        origin_history=origin_history.to(device).float()
        origin_future=origin_future.to(device).float()
        batch_history_mark=batch_history_mark.to(device).float()
        batch_future_mark=batch_future_mark.to(device).float()


        batch_history_sliding_cov, batch_history_sliding_cov_trig = sliding_cov(
            batch_x=batch_history, window_size=window_size, pad_mode=pad_mode
        )
        batch_future_sliding_cov, batch_future_sliding_cov_trig = sliding_cov(
            batch_x=batch_future, window_size=window_size, pad_mode=pad_mode
        )

        batch_his_xxT, batch_his_xxT_trig = batch_history_sliding_cov.detach(), batch_history_sliding_cov_trig.detach()
        batch_fur_xxT, batch_fur_xxT_trig = batch_future_sliding_cov.detach(), batch_future_sliding_cov_trig.detach()


        dec_inp_pred = torch.zeros(
            [batch_history.size(0), pred_len, num_features + int(num_features*(num_features+1)/2)]
        ).to(device)
        dec_inp_label = torch.cat([batch_history[:, -label_len :, :].to(device),batch_his_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)  # [1, label_len+pred_len, N+n_up_trig]

        batch_future_mark_input = torch.concat([batch_history_mark[:, -label_len:, :], batch_future_mark], dim=1)

        miu_pred, sigma_pred, _ = model_conditional_mean(
                x_enc = batch_history,
                x_enc_xxT_trig = batch_his_xxT_trig,
                x_mark_enc = batch_history_mark.to(device).float(),
                x_dec = dec_inp,
                x_mark_dec = batch_future_mark_input.to(device).float(),
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                future_mixup_weight=1,
                batch_y=None, batch_yyT_trig=None
            )
        neural_cov_last_batch = compute_neural_cov(sigma_pred, miu_pred)
        future_cw, cov_sqrt_inv, cov_sqrt = whiten_sequence_fast(batch_future, 
                                    conditional_mean=miu_pred.detach(), 
                                    cov_matrix=neural_cov_last_batch.detach(), eps=eps_eign_min, verbose = False)
        future_cw_train[i] = {'batch_history': batch_history.detach(),
                                'batch_future': batch_future.detach(),
                                'batch_history_mark': batch_history_mark.detach(),
                                'batch_future_mark': batch_future_mark.detach(),
                                'batch_future_cw': future_cw.detach(),
                                'batch_cov_sqrt_inv': cov_sqrt_inv.detach(),
                                'cov_sqrt': cov_sqrt.detach(),
                                'miu_pred': miu_pred.detach(),
                                }




    future_cw_val = {}
    # with torch.inference_mode():
    model_conditional_mean.eval()
    for i, (batch_history,
            batch_future,
            origin_history,
            origin_future,
            batch_history_mark,
            batch_future_mark,
            ) in enumerate(dataloader_cw.val_loader):
        # print(f'{i}-th batch is whitened')
        batch_history=batch_history.to(device).float()
        batch_future=batch_future.to(device).float()
        origin_history=origin_history.to(device).float()
        origin_future=origin_future.to(device).float()
        batch_history_mark=batch_history_mark.to(device).float()
        batch_future_mark=batch_future_mark.to(device).float()


        batch_history_sliding_cov, batch_history_sliding_cov_trig = sliding_cov(
            batch_x=batch_history, window_size=window_size, pad_mode=pad_mode
        )
        batch_future_sliding_cov, batch_future_sliding_cov_trig = sliding_cov(
            batch_x=batch_future, window_size=window_size, pad_mode=pad_mode
        )

        batch_his_xxT, batch_his_xxT_trig = batch_history_sliding_cov.detach(), batch_history_sliding_cov_trig.detach()
        batch_fur_xxT, batch_fur_xxT_trig = batch_future_sliding_cov.detach(), batch_future_sliding_cov_trig.detach()


        dec_inp_pred = torch.zeros(
            [batch_history.size(0), pred_len, num_features + int(num_features*(num_features+1)/2)]
        ).to(device)

        dec_inp_label = torch.cat([batch_history[:, -label_len :, :].to(device),batch_his_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
        # print(dec_inp_label.shape, dec_inp_pred.shape)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)  # [1, label_len+pred_len, N+n_up_trig]

        batch_future_mark_input = torch.concat([batch_history_mark[:, -label_len:, :], batch_future_mark], dim=1)

        # print(batch_history.shape, batch_his_xxT_trig.shape, batch_history_mark.shape, dec_inp.shape, batch_x_mark.shape, )
        miu_pred, sigma_pred, _ = model_conditional_mean(
                x_enc = batch_history,
                x_enc_xxT_trig = batch_his_xxT_trig,
                x_mark_enc = batch_history_mark.to(device).float(),
                x_dec = dec_inp,
                x_mark_dec = batch_future_mark_input.to(device).float(),
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                future_mixup_weight=1,
                batch_y=None, batch_yyT_trig=None
            )
        neural_cov_last_batch = compute_neural_cov(sigma_pred, miu_pred)
        future_cw, cov_sqrt_inv, cov_sqrt = whiten_sequence_fast(batch_future, 
                                    conditional_mean=miu_pred.detach(), 
                                    cov_matrix=neural_cov_last_batch.detach(), eps=eps_eign_min, verbose = False)
        future_cw_val[i] = {'batch_history': batch_history.detach(),
                                'batch_future': batch_future.detach(),
                                'batch_history_mark': batch_history_mark.detach(),
                                'batch_future_mark': batch_future_mark.detach(),
                                'batch_future_cw': future_cw.detach(),
                                'batch_cov_sqrt_inv': cov_sqrt_inv.detach(),
                                'cov_sqrt': cov_sqrt.detach(),
                                'miu_pred': miu_pred.detach(),
                                }
        # future_cw_val[i] = [batch_future.detach(), future_cw.detach(), cov_sqrt_inv.detach(), cov_sqrt.detach(), miu_pred.detach()]




    future_cw_test = {}
    # with torch.inference_mode():
    model_conditional_mean.eval()
    for i, (batch_history,
            batch_future,
            origin_history,
            origin_future,
            batch_history_mark,
            batch_future_mark,
            ) in enumerate(dataloader_cw.test_loader):
        # print(f'{i}-th batch is whitened')
        batch_history=batch_history.to(device).float()
        batch_future=batch_future.to(device).float()
        origin_history=origin_history.to(device).float()
        origin_future=origin_future.to(device).float()
        batch_history_mark=batch_history_mark.to(device).float()
        batch_future_mark=batch_future_mark.to(device).float()


        batch_history_sliding_cov, batch_history_sliding_cov_trig = sliding_cov(
            batch_x=batch_history, window_size=window_size, pad_mode=pad_mode
        )
        batch_future_sliding_cov, batch_future_sliding_cov_trig = sliding_cov(
            batch_x=batch_future, window_size=window_size, pad_mode=pad_mode
        )


        batch_his_xxT, batch_his_xxT_trig = batch_history_sliding_cov.detach(), batch_history_sliding_cov_trig.detach()
        batch_fur_xxT, batch_fur_xxT_trig = batch_future_sliding_cov.detach(), batch_future_sliding_cov_trig.detach()


        dec_inp_pred = torch.zeros(
            [batch_history.size(0), pred_len, num_features + int(num_features*(num_features+1)/2)]
        ).to(device)

        dec_inp_label = torch.cat([batch_history[:, -label_len :, :].to(device),batch_his_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
        # print(dec_inp_label.shape, dec_inp_pred.shape)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)  # [1, label_len+pred_len, N+n_up_trig]

        batch_future_mark_input = torch.concat([batch_history_mark[:, -label_len:, :], batch_future_mark], dim=1)

        # print(batch_history.shape, batch_his_xxT_trig.shape, batch_history_mark.shape, dec_inp.shape, batch_x_mark.shape, )
        miu_pred, sigma_pred, _ = model_conditional_mean(
                x_enc = batch_history,
                x_enc_xxT_trig = batch_his_xxT_trig,
                x_mark_enc = batch_history_mark.to(device).float(),
                x_dec = dec_inp,
                x_mark_dec = batch_future_mark_input.to(device).float(),
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                future_mixup_weight=1,
                batch_y=None, batch_yyT_trig=None
            )
        neural_cov_last_batch = compute_neural_cov(sigma_pred, miu_pred)
        future_cw, cov_sqrt_inv, cov_sqrt = whiten_sequence_fast(batch_future, 
                                    conditional_mean=miu_pred.detach(), 
                                    cov_matrix=neural_cov_last_batch.detach(), eps=eps_eign_min, verbose = False)
        future_cw_test[i] = {'batch_history': batch_history.detach(),
                                'batch_future': batch_future.detach(),
                                'batch_history_mark': batch_history_mark.detach(),
                                'batch_future_mark': batch_future_mark.detach(),
                                'batch_future_cw': future_cw.detach(),
                                'batch_cov_sqrt_inv': cov_sqrt_inv.detach(),
                                'cov_sqrt': cov_sqrt.detach(),
                                'miu_pred': miu_pred.detach(),
                                }
        # future_cw_test[i] = [batch_future.detach(), future_cw.detach(), cov_sqrt_inv.detach(), cov_sqrt.detach(), miu_pred.detach()]





    # ##### Metrics


    from torchmetrics import Metric
    import CRPS.CRPS as pscore  # Assuming `pscore` is the function to compute CRPS
    from concurrent.futures import ProcessPoolExecutor

    class CRPS(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")


            # self.executor = ProcessPoolExecutor()
            

        def update(self, pred: torch.Tensor, true: torch.Tensor):
            """
            Args:
                pred: Tensor of predicted distributions, shape (B, O, N, S).
                true: Tensor of true values, shape (B, O, N,).
            """
            def compute_crps(i):
                return pscore(pred_np[i], true_np[i]).compute()[0]

            pred = pred.view(-1, pred.shape[3])  # Reshape to (B * O * N, S)
            true = true.view(-1)  # Reshape to (B * O * N,)
            
            pred_np = pred.cpu().numpy()
            true_np = true.cpu().numpy()

            # crps_sum = sum(self.executor.map(compute_crps, range(len(true_np))))
            
            crps_sum = 0.0
            for i in range(len(true_np)):
                res = pscore(pred_np[i], true_np[i]).compute()
                crps_sum += res[0]

            self.total_crps += torch.tensor(crps_sum).to(self.device)
            self.total_samples += pred.size(0)

        def compute(self):
            return self.total_crps / self.total_samples

    class CRPSSum(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, pred: torch.Tensor, true: torch.Tensor):
            """
            Args:
                pred: Tensor of predicted distributions, shape (N, S).
                true: Tensor of true values, shape (N,).
            """
            
            pred = pred.sum(dim=2)
            true = true.sum(dim=2)
            

            pred = pred.view(-1, pred.shape[2])  # Reshape to (B * O , S)
            true = true.view(-1)  # Reshape to (B * O,)

            
            pred_np = pred.cpu().numpy()
            true_np = true.cpu().numpy()

            crps_sum = 0.0
            for i in range(len(true_np)):
                res = pscore(pred_np[i], true_np[i]).compute()
                crps_sum += res[0]

            self.total_crps += torch.tensor(crps_sum).to(self.device)
            self.total_samples += pred.size(0)

        def compute(self):
            return self.total_crps / self.total_samples
        

    class PICP(Metric):
        def __init__(self, low_percentile: int = 5, high_percentile: int = 95, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.low_percentile = low_percentile
            self.high_percentile = high_percentile
            self.add_state("coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, all_gen_y: torch.Tensor, y_true: torch.Tensor):
            # Reshape to (B * O * N, S)
            all_gen_y = all_gen_y.view(-1, all_gen_y.shape[3]).cpu()
            y_true = y_true.view(-1).cpu()  # Reshape to (B * O * N,)

            # Compute the low and high percentiles using torch.quantile
            low, high = self.low_percentile, self.high_percentile
            CI_y_pred = torch.quantile(all_gen_y, torch.tensor([low / 100.0, high / 100.0]).float(), dim=1)
            
            # Determine whether the true values are within the prediction intervals
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            
            coverage = y_in_range.float().mean()
            self.coverage += coverage.to(self.device)
            self.total_samples += y_true.size(0)

        def compute(self):
            return self.coverage / self.total_samples
        

    class ProbMAE(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("total_mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, pred: torch.Tensor, true: torch.Tensor):
            """
            Args:
                pred: Tensor of predicted distributions, shape (B, O, N, S).
                true: Tensor of true values, shape (B, O, N).
            """
            # Compute mean along S-axis
            pred_mean = pred.mean(dim=-1)  # Shape: (B, O, N)

            # Ensure the true tensor matches the shape
            assert true.shape == pred_mean.shape, "Shapes of true values and pred_mean must match"

            # Compute absolute error
            absolute_error = torch.abs(pred_mean - true)

            # Sum errors and count total samples
            self.total_mae += absolute_error.sum()
            self.total_samples += absolute_error.numel()

        def compute(self):
            # Compute mean absolute error
            return self.total_mae / self.total_samples
        
    class ProbMSE(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("total_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, pred: torch.Tensor, true: torch.Tensor):
            """
            Args:
                pred: Tensor of predicted distributions, shape (B, O, N, S).
                true: Tensor of true values, shape (B, O, N).
            """
            # Compute mean along S-axis
            pred_mean = pred.mean(dim=-1)  # Shape: (B, O, N)

            # Ensure the true tensor matches the shape
            assert true.shape == pred_mean.shape, "Shapes of true values and pred_mean must match"

            # Compute squared error
            squared_error = (pred_mean - true) ** 2

            # Sum errors and count total samples
            self.total_mse += squared_error.sum()
            self.total_samples += squared_error.numel()

        def compute(self):
            # Compute mean squared error
            return self.total_mse / self.total_samples


    class ProbRMSE(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("total_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, pred: torch.Tensor, true: torch.Tensor):
            """
            Args:
                pred: Tensor of predicted distributions, shape (B, O, N, S).
                true: Tensor of true values, shape (B, O, N).
            """
            # Compute mean along S-axis
            pred_mean = pred.mean(dim=-1)  # Shape: (B, O, N)

            # Ensure the true tensor matches the shape
            assert true.shape == pred_mean.shape, "Shapes of true values and pred_mean must match"

            # Compute squared error
            squared_error = (pred_mean - true) ** 2

            # Sum errors and count total samples
            self.total_mse += squared_error.sum()
            self.total_samples += squared_error.numel()

        def compute(self):
            # Compute root mean squared error
            return torch.sqrt(self.total_mse / self.total_samples)
        

    class QICE(Metric):
        def __init__(self, n_bins: int = 10, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.n_bins = n_bins
            # Add states for each quantile's coverage ratio
            self.add_state("quantile_bin_counts", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
            
        def update(self, preds: torch.Tensor, targets: torch.Tensor):
            """
            Update the metric with the predictions and targets.
            Args:
                preds: Tensor of shape (N, S) containing generated predictions
                targets: Tensor of shape (N, 1) containing ground truth values
            """
            # print(preds[0, :, 0, :], targets[0, :, 0])
            
            preds = preds.view(-1, preds.size(3))  # Reshape to (B * O * N, S)
            targets = targets.view(-1)  # Reshape to (B * O * N,)

            preds_np = preds.cpu().numpy()  # Shape (N, S)
            targets_np = targets.cpu().numpy().T  # Shape (1, N)
            
            # Generate quantiles based on the number of bins
            quantile_list = np.arange(self.n_bins + 1) * (100 / self.n_bins)
            
            # Calculate the quantiles for the predicted values
            y_pred_quantiles = np.percentile(preds_np, q=quantile_list, axis=1)  # Shape (n_bins+1, N)
            
            # Calculate which quantile interval the true target belongs to
            quantile_membership_array = ((targets_np - y_pred_quantiles) > 0).astype(int)  # Shape (n_bins+1, N)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)  # Shape (N,)
            
            # Count the number of targets in each bin
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(self.n_bins + 2)]  # Shape (n_bins+2,)
            )
            print(y_true_quantile_bin_count)
            # Combine outliers into the first and last bins
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]  # Exclude first and last bin
            
            # Update the quantile bin counts for each update
            self.quantile_bin_counts += torch.tensor(y_true_quantile_bin_count_).to(self.device)
            self.total_samples += preds.size(0)
            
        def compute(self):
            """
            Compute the QICE score (geometric mean of coverage ratios).
            Returns:
                The QICE score as a float.
            """
            # Normalize the counts by the total number of samples
            
            
            y_true_ratio_by_bin = self.quantile_bin_counts.float() / self.total_samples.item()
            # print(self.total_samples,self.quantile_bin_counts )
            # print(y_true_ratio_by_bin.shape, torch.sum(y_true_ratio_by_bin),  torch.abs(
            #     torch.sum(y_true_ratio_by_bin) - 1))
            assert torch.abs(
                torch.sum(y_true_ratio_by_bin) - 1) < 1e-5, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = torch.abs(torch.ones(self.n_bins) / self.n_bins - y_true_ratio_by_bin).mean()
            return qice_coverage_ratio


    from ts2vec.ts2vec import TS2Vec
    import scipy
    def cacf_torch(x, max_lag = 1, dim=(0, 1)):
        def get_lower_triangular_indices(n):
            return [list(x) for x in torch.tril_indices(n, n)]

        ind = get_lower_triangular_indices(x.shape[2])
        x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
        x_l = x[..., ind[0]]
        x_r = x[..., ind[1]]
        cacf_list = list()
        for i in range(max_lag):
            y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
            cacf_i = torch.mean(y, (1))
            cacf_list.append(cacf_i)
        cacf = torch.cat(cacf_list, 1)
        return cacf.reshape(cacf.shape[0], -1, len(ind[0]))





    def calculate_fid(act1, act2):
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def Context_FID(ori_data, generated_data):
        model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=8, lr=0.001, output_dims=320,
                    max_train_length=3000)
        model.fit(ori_data, verbose=False)
        ori_represenation = model.encode(ori_data, encoding_window='full_series')
        gen_represenation = model.encode(generated_data, encoding_window='full_series')
        idx = np.random.permutation(ori_data.shape[0])
        ori_represenation = ori_represenation[idx] # [idx] is from https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/context_fid.py. actually i dont know why they use [idx]
        gen_represenation = gen_represenation[idx]
        results = calculate_fid(ori_represenation, gen_represenation)
        return results



    # most of the following codes are from https://github.com/wwy155/NsDiff 
    def wv_sigma(x_enc, window_size):
        """
        Compute the variance over a sliding window along the T dimension.

        For each time step t, the variance is calculated over a window of size `window_size`
        centered around t. For even window sizes, the window is asymmetrically padded to maintain
        the same output length as the input.

        Args:
            x_enc (Tensor): Input tensor of shape (B, T, N)
            window_size (int): Size of the sliding window

        Returns:
            sigma (Tensor): Variance tensor of shape (B, T, N)
        """
        B, T, N = x_enc.shape
        if window_size % 2 == 0:
            pad_left = window_size // 2
            pad_right = window_size // 2 - 1
        else:
            pad_left = pad_right = window_size // 2
        x_padded = F.pad(x_enc, (0, 0, pad_left, pad_right), mode='replicate')
        windows = x_padded.unfold(dimension=1, size=window_size, step=1)

        sigma = windows.var(dim=3, unbiased=False)  # Shape: (B, T, N)

        return sigma


    def wv_sigma_trailing(x_enc, window_size, discard_rep=False):
        """
        Compute the variance over a trailing window for each time step.

        For each time step t, the variance is calculated over the window [t - window_size, t - 1].

        Args:
            x_enc (Tensor): Input tensor of shape (B, T, N)
            window_size (int): Size of the trailing window

        Returns:
            sigma (Tensor): Variance tensor of shape (B, T, N)
        """
        if not isinstance(x_enc, torch.Tensor):
            raise TypeError("x_enc must be a torch.Tensor")

        if x_enc.dim() != 3:
            raise ValueError("x_enc must be a 3D tensor with shape (B, T, N)")

        B, T, N = x_enc.shape

        if window_size < 1 or window_size > T:
            raise ValueError(f"window_size must be between 1 and T (got window_size={window_size}, T={T})")

        # Pad the beginning of the T dimension with window_size elements
        # This ensures that for the first window_size time steps, we have enough elements
        # Use 'replicate' padding to repeat the first time step
        if not discard_rep:
            x_enc = F.pad(x_enc, (0, 0, window_size, 0), mode='replicate')  # Shape: (B, T + window_size, N)

        # Create sliding windows of size window_size along the T dimension
        # Each window will cover [t - window_size, t - 1] after padding
        # The resulting shape will be (B, T, window_size, N)
        windows = x_enc.unfold(1, window_size, 1) 

        # Compute variance across the window dimension (dim=2)
        sigma = windows.var(dim=3, unbiased=False)  # Shape: (B, T, N)
        return sigma





    def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == "linear":
            betas = torch.linspace(start, end, num_timesteps)
        elif schedule == "const":
            betas = end * torch.ones(num_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
        elif schedule == "jsd":
            betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == "cosine" or schedule == "cosine_reverse":
            max_beta = 0.999
            cosine_s = 0.008
            betas = torch.tensor(
                [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                        math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
                range(num_timesteps)])
            if schedule == "cosine_reverse":
                betas = betas.flip(0)  # starts at max_beta then decreases fast
        elif schedule == "cosine_anneal":
            betas = torch.tensor(
                [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
                range(num_timesteps)])
        return betas

    def extract(input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def cal_forward_noise(betas_tiled, betas_bar, gx, y_sigma, t):
        b_bar_t =  extract(betas_bar, t, gx)
        b_tilded_t =  extract(betas_tiled, t, gx)
        
        noise = (b_bar_t - b_tilded_t)*gx + b_tilded_t*y_sigma
        assert (noise >= 0).all()
        return noise

    def cal_forward_noise_full(betas_tiled, full_gt, gx, y_sigma, t):
        full_gt_t =  extract(full_gt, t, gx)
        b_tilded_t =  extract(betas_tiled, t, gx)
        
        noise = (full_gt_t)*gx + b_tilded_t*y_sigma
        assert (noise >= 0).all()
        return noise

    def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
        """
        Input:
        y: original data, y_0.(B, T, N)
        y_0_hat: prediction of pre-trained guidance model. (B, T, N)
        alphas_bar_sqrt: square root of cumulative product of alphas at timestep t.(T, N)
        one_minus_alphas_bar_sqrt: square root of cumulative product of (1 - alphas) at timestep t.(T, N)
        t: current timestep, a scalar tensor. (B, 1)
        noise: optional noise tensor, if None, will be sampled from standard normal distribution. (B, T, N)
        Returns:
            y_t: sampled data at timestep t, q(y_t | y_0, x).(B, T, N)
        """
        if noise is None:
            noise = torch.randn_like(y).to(y.device)
        sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        # q(y_t | y_0, x)
        y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + noise
        return y_t

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def compute_gx_term(alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.float()
        n = alpha.shape[0]
        gx_term = torch.zeros_like(alpha)  
        for t in range(n):
            slice_t = alpha[:t+1].flip(dims=[0]) # at, at-1, at-2, a1
            cprod = torch.cat([torch.tensor([1]).to(slice_t.device), torch.cumprod(slice_t, dim=0)])
            cprod = cprod[:-1] * ((1 - slice_t)**2) # at^2, at-1^2*at, at-1*at-2^2*at, ...
            gx_term[t] = cprod.sum()
        return gx_term

    # alpha_tilde
    def compute_tilde_alpha(alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.float()
        n = alpha.shape[0]
        tilde_alpha = torch.zeros_like(alpha)  
        
        for t in range(n):
            slice_t = alpha[:t+1].flip(dims=[0])
            cprod = torch.cumprod(slice_t, dim=0)
            tilde_alpha[t] = cprod.sum()
        return tilde_alpha

    # alhpa_hat
    def compute_hat_alpha(alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.float()
        n = alpha.shape[0]
        hat_alpha = torch.zeros_like(alpha)  
        for t in range(n):
            slice_t = alpha[:t+1].flip(dims=[0]) # at, at-1, at-2, ...
            cprod = torch.cumprod(slice_t, dim=0) # at, at-1*at, at-1*at-2*at, ...
            cprod = cprod * slice_t # at^2, at-1^2*at, at-1*at-2^2*at, ...
            hat_alpha[t] = cprod.sum()
        return hat_alpha



    def cal_sigma12(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
        at = extract(alphas, t, gx)
        at_bar = extract(alphas_cumprod, t, gx)
        # at_bar_prev = extract(alpha_bar_prev, t, gx)
        at_tilde = extract(alphas_cumprod_sum, t, gx)
        b_tilde_m_1 = extract(betas_tiled_m_1, t, gx)
        b_bar_m_1 = extract(betas_bar_m_1, t, gx)
        # at_tilde_prev = extract(alphas_cumprod_sum_prev, t, gx)

        Sigma_1 = (1 - at)**2*gx + at*(1 - at)*y_sigma 
        Sigma_2 = (b_bar_m_1 - b_tilde_m_1)*gx + b_tilde_m_1*y_sigma
        # sigma_tilde = (Sigma_1*Sigma_2)/(at * Sigma_2 + Sigma_1)
        # # mu_tilde = (Sigma_1*Sigma_2)/(at * Sigma_2 + Sigma_1)
        # Sigma_1 = 1 - at
        # Sigma_2 = 1 - at_bar_prev
        return at, at_bar, at_tilde, Sigma_1, Sigma_2

    def cal_sigma_tilde(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
        at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t)
        sigma_tilde = (Sigma_1*Sigma_2)/(at * Sigma_2 + Sigma_1)
        return sigma_tilde

    def calc_gammas(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
        at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t)
        
        alpha_bar_t_m_1 = extract(alpha_bar_prev, t, gx)
        sqrt_alpha_t = at.sqrt()
        sqrt_alpha_bar_t_m_1 = alpha_bar_t_m_1.sqrt()
        
        at_s1_s2 = at*Sigma_2 + Sigma_1
        
        gamma_0 = sqrt_alpha_bar_t_m_1*Sigma_1/at_s1_s2
        gamma_1 = sqrt_alpha_t*Sigma_2/at_s1_s2
        gamma_2 = ((sqrt_alpha_t*(at - 1))*Sigma_2 + (1 - sqrt_alpha_bar_t_m_1)*Sigma_1)/at_s1_s2
        return gamma_0, gamma_1, gamma_2

    def p_sample(model, x, x_mark, y, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_all, betas_bar_all, betas_tiled_m_1_all, betas_bar_m_1_all):
        """
        Input:
        x: input time series data. (B, T, N)
        x_mark: time marks. (B, T)
        y: sampled y at time step t, y_t. (B, T, N)
        y_0_hat: prediction of pre-trained guidance model. (B, T, N)
        y_T_mean: mean of prior distribution at timestep T. (B, T, N)
        gx: condition variance. (B, T, N)

        Returns:
        y_t_m_1: sampled y at time step t-1, p(y_{t-1} | y_t, x). (B, T, N)

        We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
            guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
            in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
        """
        device = next(model.parameters()).device
        t = torch.tensor([t]).to(device)

        eps_theta, sigma_theta = model(x, x_mark, y, y_0_hat, gx, t)
        
        eps_theta = eps_theta.to(device).detach()
        sigma_theta = sigma_theta.to(device).detach()
        
        z =  torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
        alpha_t = extract(alphas, t, y)
        
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
        
        betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
        betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
        betas_tiled = extract(betas_tiled_all, t, y)
        betas_bar = extract(betas_bar_all, t, y)

        lambda_0 = alpha_t*(1 - alpha_t)*betas_tiled_m_1
        lambda_1 = ((1 - alpha_t)**2*betas_tiled_m_1 + alpha_t*(1 - alpha_t)*(betas_bar_m_1 - betas_tiled_m_1))*gx - sigma_theta*(alpha_t*betas_tiled_m_1 + alpha_t*(1 - alpha_t))
        lambda_2 = gx**2*(1 - alpha_t)**2*(betas_bar_m_1 - betas_tiled_m_1) - sigma_theta*gx*(alpha_t*betas_bar_m_1 - alpha_t*betas_tiled_m_1 + (1 - alpha_t)**2)
        sigma_y0_hat = (-lambda_1 + ((lambda_1)**2 - 4*lambda_0*lambda_2).sqrt()  )/(2*lambda_0)
        
        noise = (betas_bar - betas_tiled)*gx + betas_tiled*sigma_y0_hat
        
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta*torch.sqrt(noise))
        gamma_0, gamma_1, gamma_2 = calc_gammas(alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1_all, betas_bar_m_1_all, gx, sigma_y0_hat, t)
        
        y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
        y_t_m_1 = y_t_m_1_hat.to(device) + torch.sqrt(sigma_theta) *z.to(device)
        return y_t_m_1


    def p_sample_pe(model, x, x_mark, y, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_all, betas_bar_all, betas_tiled_m_1_all, betas_bar_m_1_all):

        device = next(model.parameters()).device
        t = torch.tensor([t]).to(device)

        eps_theta, sigma_theta = model(x, x_mark, y, y_0_hat, gx, t)
        
        eps_theta = eps_theta.to(device).detach()
        sigma_theta = sigma_theta.to(device).detach()
        
        z =  torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
        alpha_t = extract(alphas, t, y)
        
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
        
        betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
        betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
        betas_tiled = extract(betas_tiled_all, t, y)
        betas_bar = extract(betas_bar_all, t, y)

        sigma_y0_hat = gx #(-lambda_1 + ((lambda_1)**2 - 4*lambda_0*lambda_2).sqrt()  )/(2*lambda_0)

        noise = (betas_bar)*gx
        
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta*torch.sqrt(noise))
        # posterior mean
        gamma_0, gamma_1, gamma_2 = calc_gammas(alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1_all, betas_bar_m_1_all, gx, gx, t)
        y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
        # posterior variance
        y_t_m_1 = y_t_m_1_hat.to(device) + torch.sqrt(sigma_theta) *z.to(device)
        return y_t_m_1

    def p_sample_t_1to0(model, x, x_mark, y, y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt,alphas,alphas_cumprod,alphas_cumprod_sum,alpha_bar_prev, alphas_cumprod_sum_prev,betas_tiled_all, betas_bar_all, betas_tiled_m_1_all, betas_bar_m_1_all):
        device = next(model.parameters()).device
        t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        eps_theta, sigma_theta = model(x, x_mark, y, y_0_hat, gx, t)
        
        # at_tilde = extract(alphas_cumprod_sum, t, gx)
        
        eps_theta = eps_theta.to(device).detach()
        sigma_theta = sigma_theta.to(device).detach()
        alpha_t = extract(alphas, t, y)
        
        betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
        betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
        betas_tiled = extract(betas_tiled_all, t, y)
        betas_bar = extract(betas_bar_all, t, y)

        # estimate Sigma Y0
        lambda_0 = alpha_t*(1 - alpha_t)*betas_tiled_m_1
        lambda_1 = ((1 - alpha_t)**2*betas_tiled_m_1 + alpha_t*(1 - alpha_t)*(betas_bar_m_1 - betas_tiled_m_1))*gx - sigma_theta*(alpha_t*betas_tiled_m_1 + alpha_t*(1 - alpha_t))
        lambda_2 = gx**2*(1 - alpha_t)**2*(betas_bar_m_1 - betas_tiled_m_1) - sigma_theta*gx*(alpha_t*betas_bar_m_1 - alpha_t*betas_tiled_m_1 + (1 - alpha_t)**2)
        sigma_y0_hat = (-lambda_1 + ((lambda_1)**2 - 4*lambda_0*lambda_2).sqrt()  )/(2*lambda_0)
        noise = (betas_bar - betas_tiled)*gx + betas_tiled*sigma_y0_hat
        
        # y_0 reparameterization
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise))
        y_t_m_1 = y_0_reparam.to(device)
        return y_t_m_1

    def p_sample_t_1to0_pe(model, x, x_mark, y, y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt,alphas,alphas_cumprod,alphas_cumprod_sum,alpha_bar_prev, alphas_cumprod_sum_prev,betas_tiled_all, betas_bar_all, betas_tiled_m_1_all, betas_bar_m_1_all):
        device = next(model.parameters()).device
        t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        eps_theta, sigma_theta = model(x, x_mark, y, y_0_hat, gx, t)
        
        # at_tilde = extract(alphas_cumprod_sum, t, gx)
        
        eps_theta = eps_theta.to(device).detach()
        sigma_theta = sigma_theta.to(device).detach()
        alpha_t = extract(alphas, t, y)
        
        betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
        betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
        betas_tiled = extract(betas_tiled_all, t, y)
        betas_bar = extract(betas_bar_all, t, y)

        # estimate Sigma Y0
        sigma_y0_hat = gx #(-lambda_1 + ((lambda_1)**2 - 4*lambda_0*lambda_2).sqrt()  )/(2*lambda_0)
        noise = (betas_bar)*gx
        
        
        # y_0 reparameterization
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise))
        y_t_m_1 = y_0_reparam.to(device)
        return y_t_m_1



    def p_sample_loop(model, x, x_mark, y_0_hat, gx, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1):
        device = next(model.parameters()).device
        z = torch.randn_like(y_T_mean).to(device) # sample 
        cur_y = torch.sqrt(gx) * z + y_T_mean  # sample y_T
        y_p_seq = [cur_y]
        for t in reversed(range(1, n_steps)):  # t from T to 2
            y_t = cur_y
            cur_y = p_sample(model, x, x_mark, y_t, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)  # y_{t-1}
            y_p_seq.append(cur_y)
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(model, x, x_mark, y_p_seq[-1], y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt,alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)
        y_p_seq.append(y_0)
        return y_p_seq 

    def p_sample_loop_pe(model, x, x_mark, y_0_hat, gx, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1):
        device = next(model.parameters()).device
        z = torch.randn_like(y_T_mean).to(device) # sample 
        cur_y = torch.sqrt(gx) * z + y_T_mean  # sample y_T
        
        y_p_seq = [cur_y]
        for t in reversed(range(1, n_steps)):  # t from T to 2
            y_t = cur_y
            cur_y = p_sample_pe(model, x, x_mark, y_t, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)  # y_{t-1}
            y_p_seq.append(cur_y)
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0_pe(model, x, x_mark, y_p_seq[-1], y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt,alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)
        y_p_seq.append(y_0)
        return y_p_seq

    def kld(y1, y2, grid=(-20, 20), num_grid=400):
        y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()
        p_y1, _ = np.histogram(y1, bins=num_grid, range=[grid[0], grid[1]], density=True)
        p_y1 += 1e-7
        p_y2, _ = np.histogram(y2, bins=num_grid, range=[grid[0], grid[1]], density=True)
        p_y2 += 1e-7
        return (p_y1 * np.log(p_y1 / p_y2)).sum()





    import torch
    import torch.nn as nn
    from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
    from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
    from torch_timeseries.nn.embedding import DataEmbedding


    class Projector(nn.Module):
        '''
        MLP to learn the De-stationary factors
        '''
        def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
            super(Projector, self).__init__()

            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                        padding_mode='circular', bias=False)

            layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
            for i in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

            layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
            self.backbone = nn.Sequential(*layers)


        def forward(self, x, stats):

            batch_size = x.shape[0]
            x = self.series_conv(x)  # B x 1 x D
            x = torch.cat([x, stats], dim=1)  # B x 2 x D
            x = x.view(batch_size, -1)  # B x 2D
            y = self.backbone(x)  # B x output_dim

            return y


    class ns_Transformer(nn.Module):
        """
        Non-stationary Transformer
        """
        def __init__(self, configs):
            super(ns_Transformer, self).__init__()
            self.pred_len = configs.pred_len 
            self.seq_len = configs.seq_len 
            self.label_len = configs.label_len 
            self.output_attention = configs.output_attention 

            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout) 
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout) 

            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

            self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                        hidden_layers=configs.p_hidden_layers, output_dim=1)
            self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                        hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                        output_dim=configs.seq_len)

            self.z_mean = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, configs.d_model)
            )
            self.z_logvar = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, configs.d_model)
            )

            self.z_out = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, configs.d_model)
            )

        def KL_loss_normal(self, posterior_mean, posterior_logvar):
            KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                                torch.exp(posterior_logvar), dim=1)
            return torch.mean(KL)
        

        def reparameterize(self, posterior_mean, posterior_logvar):
            posterior_var = posterior_logvar.exp()
            # take sample
            if self.training:
                posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
                posterior_var = posterior_var.repeat(100, 1, 1, 1)
                eps = torch.zeros_like(posterior_var).normal_()
                z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
                z = z.mean(0)
            else:
                z = posterior_mean
            # z = posterior_mean
            return z

        def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                    enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
            """
            x_enc (Tensor): Encoded input sequence of shape (B, seq_len, enc_in)
            x_mark_enc (Tensor): Encoded input time features of shape (B, seq_len, d)
            x_dec (Tensor): Decoded input sequence of shape (B, seq_len, dec_in)
            x_mark_dec (Tensor): Decoded input time features of shape (B, seq_len, dec_in)
            """
            x_raw = x_enc.clone().detach()

            # Normalization
            mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
            x_enc = x_enc - mean_enc
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
            x_enc = x_enc / std_enc

            x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                                dim=1).to(x_enc.device).clone()

            tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
            delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

            dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

            # this is very important. DO NOT do denormalization at here!!!
            # dec_out = dec_out * std_enc + mean_enc

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:

                return dec_out[:, -self.pred_len:, :], dec_out  # [B, L, D]




    class SigmaEstimation(nn.Module):
        """
        Non-stationary Transformer
        """
        def __init__(self, seq_len, pred_len, enc_in, hidden_size=512, kernel_size=24):
            super(SigmaEstimation, self).__init__()
            self.pred_len = pred_len 
            self.seq_len = seq_len 
            self.enc_in = enc_in 
            self.hidden_size = hidden_size # hidden layer size
            
            
            # Define 2-layer MLP for predicting future sigmas
            self.mlp = nn.Sequential(
                nn.Linear(seq_len -  kernel_size, hidden_size),
                # nn.ReLU(),
                # nn.Linear(hidden_size, hidden_size),  # Output size should match enc_in
                # nn.ReLU(),
                nn.Linear(hidden_size, pred_len)  # Output size should match enc_in
            )
            for p in self.mlp.parameters():
                p.data.zero_()
            
            self.kernel_size = kernel_size  
            self.padding = self.kernel_size // 2  

            
        def forward(self, x_enc):
            # x_enc: B, seq_len, N
            # return sigmas: B, pred_len, N
            B, T, N = x_enc.shape
            
            sigma = wv_sigma_trailing(x_enc, self.kernel_size, discard_rep=True)

            sigma = sigma[:, -(T - self.kernel_size):, :] + 1e-5

            pred_sigma = self.mlp(sigma.permute(0, 2, 1))  # (B, T, N) -> (B, N, T)
            pred_sigma = torch.abs(pred_sigma).permute(0, 2, 1)  # (B, O, N) where O = pred_len
            
            return pred_sigma[:, -self.pred_len:, :]




    class ConditionalLinear(nn.Module):
        def __init__(self, num_in, num_out, n_steps):
            super(ConditionalLinear, self).__init__()
            self.num_out = num_out
            self.lin = nn.Linear(num_in, num_out)
            self.embed = nn.Embedding(n_steps, num_out)
            self.embed.weight.data.uniform_()

        def forward(self, x, t):
            # x: (B, T, num_in) or (B, num_in)
            # t: (B, ) or (B, 1)

            out = self.lin(x)
            gamma = self.embed(t)
            out = gamma.view(t.size()[0], -1, self.num_out) * out
            return out # (B, T, num_out) or (B, num_out)

    class ConditionalGuidedModel(nn.Module):
        def __init__(self, diff_steps, enc_in):
            super(ConditionalGuidedModel, self).__init__()
            n_steps = diff_steps + 1 
            data_dim = enc_in * 3 
            self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
            self.lin2 = ConditionalLinear(128, 128, n_steps)
            self.lin3 = ConditionalLinear(128, 128, n_steps)
            self.lin4 = nn.Linear(128, enc_in)
            self.sigma_lin = nn.Linear(128, enc_in)

        def forward(self, x, y_t, y_0_hat, g_x, t):
            # x: (B, T, N), y_t: (B, T, N), y_0_hat: (B, T, N), g_x: (B, T, N)
            # t: (B, ) or (B, 1)
            eps_pred = torch.cat((y_t, y_0_hat, g_x), dim=-1)

            eps_pred = F.softplus(self.lin1(eps_pred, t))
            eps_pred = F.softplus(self.lin2(eps_pred, t))
            eps_pred = F.softplus(self.lin3(eps_pred, t))
            # print(eps_pred.shape)
            eps_pred, sigma = self.lin4(eps_pred), F.softplus(self.sigma_lin(F.softplus(eps_pred))) # sigma 
            # print(eps_pred.shape, sigma.shape)
            return eps_pred, sigma # (B, T, N), (B, T, N)


    class NsDiff(nn.Module):
        """
        Vanilla Transformer
        """

        def __init__(self, configs, device):
            super(NsDiff, self).__init__()


            
            self.args = configs
            self.device = device

            # self.model_var_type = configs.var_type
            self.num_timesteps = configs.timesteps
            self.dataset_object = None
            betas = make_beta_schedule(schedule=configs.beta_schedule, num_timesteps=configs.timesteps,
                                    start=configs.beta_start, end=configs.beta_end)
            betas = self.betas = betas.float().to(self.device)
            self.betas_sqrt = torch.sqrt(betas)
            alphas = 1.0 - betas
            self.alphas = alphas
            self.one_minus_betas_sqrt = torch.sqrt(alphas)
            alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
            self.alphas_cumprod = alphas_cumprod
            self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
            
            self.betas_bar = 1 - self.alphas_cumprod
            # self.alphas_cumprod_sum = torch.cumsum(alphas_cumprod.flip(0), dim=0).flip(0)
            self.alphas_cumprod_sum = compute_tilde_alpha(alphas)
            
            self.alphas_tilde = self.alphas_cumprod_sum
            self.alphas_hat = compute_hat_alpha(alphas).to(self.device)
            self.betas_tilde = self.alphas_tilde  - self.alphas_hat
            self.gx_term = compute_gx_term(alphas).to(self.device) # full compute to avoid precision issue
            # import pdb;pdb.set_trace()
            assert (torch.tensor(self.betas_tilde) >= 0).all()
            # import pdb;pdb.set_trace(), ((self.betas_bar - self.betas_tilde)[((self.betas_bar - self.betas_tilde)>=0)])
            assert ((self.betas_bar - self.betas_tilde)>=0).all()
            # (self.betas_bar - self.betas_tilde)[((self.betas_bar - self.betas_tilde)>0)]
            
            
            self.betas_tilde_m_1 = torch.cat(
                [torch.ones(1, device=self.device), self.betas_tilde[:-1]], dim=0
            )
            self.betas_bar_m_1 = torch.cat(
                [torch.ones(1, device=self.device), self.betas_bar[:-1]], dim=0
            )

            
            self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
            if configs.beta_schedule == "cosine":
                self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            alphas_cumprod_prev = torch.cat(
                [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
            )
            self.alphas_cumprod_sum_prev = torch.cat(
                [torch.ones(1, device=self.device), self.alphas_cumprod_sum[:-1]], dim=0
            )

            self.alphas_cumprod_prev = alphas_cumprod_prev
            self.posterior_mean_coeff_1 = (
                    betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            self.posterior_mean_coeff_2 = (
                    torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
            )
            posterior_variance = (
                    betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            self.posterior_variance = posterior_variance
            # if self.model_var_type == "fixedlarge":
            #     self.logvar = betas.log()
            #     # torch.cat(
            #     # [posterior_variance[1:2], betas[1:]], dim=0).log()
            # elif self.model_var_type == "fixedsmall":
            #     self.logvar = posterior_variance.clamp(min=1e-20).log()

            self.tau = None  # precision fo test NLL computation

            # CATE MLP
            self.diffussion_model = ConditionalGuidedModel(configs.timesteps, configs.enc_in)

            self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                            configs.dropout)


        def forward(self, x, x_mark, y_t, y_0_hat, gx, t):
            enc_out = self.enc_embedding(x, x_mark) #  B, T, d_model
            dec_out, sigma = self.diffussion_model(enc_out, y_t, y_0_hat, gx, t)

            return dec_out, sigma


    # the experiments


    from dataclasses import dataclass, field
    import sys
    from typing import List, Dict
    import os
    import torch
    from dataclasses import dataclass, asdict, field
    from torch_timeseries.nn.embedding import freq_map
    import argparse
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
    from torch.optim import *
    from tqdm import tqdm
    from torch_timeseries.utils.model_stats import count_parameters
    from torch_timeseries.utils.reproduce import reproducible
    import time
    # import multiprocessing
    import torch.multiprocessing as mp
    from torch_timeseries.utils.parse_type import parse_type

    from torch_timeseries.utils.early_stop import EarlyStopping
    import yaml
    import numpy as np
    import torch.distributed as dist
    import torch
    from tqdm import tqdm
    import concurrent.futures
    from types import SimpleNamespace

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace


    
    class NSDiffEarlyStopping(EarlyStopping):
        def save_checkpoint(self, val_loss, model):
            """Saves model when validation loss decrease."""
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
            torch.save(model['model'].state_dict(), os.path.join(self.path, 'model.pth'))
            torch.save(model['cond_pred_model'].state_dict(),os.path.join(self.path, 'cond_pred_model.pth'))
            torch.save(model['cond_pred_model_g'].state_dict(),os.path.join(self.path, 'cond_pred_model_g.pth'))
            self.val_loss_min = val_loss
            
            
            
    def log_normal(x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
        log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
        x: (array) corresponding array containing the input
        mu: (array) corresponding array containing the mean
        var: (array) corresponding array containing the variance

        Returns:
        output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        eps = 1e-8
        if eps > 0.0:
            var = var + eps
        # return -0.5 * torch.sum(
        #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
        return 0.5 * torch.mean(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)








    data_path = "ts_datasets"
    dataset_type = "ETTh1"
    windows = 168         
    pred_len = 192         
    batch_size = 32
    epochs = 10           
    lr = 1e-3
    device = device
    num_worker = 2
    patience = 3
    scaler_type = "StandardScaler"
    optm_type = "Adam"
    train_ratio = 0.7
    test_ratio = 0.2
    invtrans_loss = False


    num_samples  = 100 
    beta_start =  0.0001
    beta_end =  0.01
    d_model =  512
    n_heads =  8
    e_layers =  2
    d_layers =  1
    d_ff =  1024
    diffusion_steps   = 20 # 20
    moving_avg =  25
    factor =  3
    distil =  True
    dropout =  0.1
    activation = 'gelu'
    k_z =  1e-2
    k_cond =  1
    d_z =  8
    CART_input_x_embed_dim = 32
    p_hidden_layers  = 2
    rolling_length  = 95
    load_pretrain  = False

    args_dict = {
        "seq_len": windows,
        "device": device,
        "pred_len": pred_len,
        "label_len": label_len,
        "features" : 'M',
        "beta_start": beta_start,
        "beta_end": beta_end,
        "enc_in" : dataset.num_features,
        "dec_in" : dataset.num_features,
        "c_out" : dataset.num_features,
        "d_model" : d_model,
        "n_heads" : n_heads,
        "e_layers" : e_layers,
        "d_layers" : d_layers,
        "d_ff" : d_ff,
        "moving_avg" : moving_avg,
        "timesteps" : diffusion_steps,
        "factor" : factor,
        "distil" : distil,
        "beta_schedule": "linear",
        "embed" : 'timeF',
        "dropout" :dropout,
        "activation" :activation,
        "output_attention" : False,
        "do_predict" :True,
        "k_z" :k_z,
        "k_cond" :k_cond,
        "p_hidden_dims" : [64, 64],
        "freq" :dataset.freq,
        "CART_input_x_embed_dim" : CART_input_x_embed_dim,
        "p_hidden_layers" : p_hidden_layers,
        "d_z" :d_z,
        "diffusion_config_dir" : "./configs/nsdiff.yml",
    }

    EPS= 1e-8
    def _process_train_batch(model,
                            cond_pred_model,
                            cond_pred_model_g,
                            batch_x, 
                            batch_y, 
                            batch_x_mark, 
                            batch_y_mark,
                            device):

        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), rolling_length) 
        y_sigma = y_sigma[:, -pred_len:, :] + EPS 
        
        batch_y_input = torch.concat([batch_x[:, -label_len:, :], batch_y], dim=1)
        batch_y_mark_input = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1) 

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), pred_len, dataset.num_features]
        ).to(device)
        dec_inp_label = batch_x[:, -label_len :, :].to(device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1) 
        

        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=model.num_timesteps, size=(n // 2 + 1,)
        ).to(device)
        t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)[:n]
        y_0_hat_batch, _ = cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        gx = cond_pred_model_g(batch_x) + 1 
        loss1 = (y_0_hat_batch - batch_y).square().mean() 
        loss2 = (torch.sqrt(gx)- torch.ones_like(y_sigma)).square().mean() 
        

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(device)

        forward_noise = cal_forward_noise(model.betas_tilde, model.betas_bar, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(model.alphas, model.alphas_cumprod, model.alphas_cumprod_sum, 
                                        model.alphas_cumprod_prev, model.alphas_cumprod_sum_prev, 
                                        model.betas_tilde_m_1, model.betas_bar_m_1, gx, y_sigma, t)

        y_t_batch = q_sample(batch_y, y_T_mean, model.alphas_bar_sqrt,
                                model.one_minus_alphas_bar_sqrt, t, noise=noise) 
        
        output, sigma_theta = model(batch_x, batch_x_mark, y_t_batch, y_0_hat_batch, gx, t)
        sigma_theta = sigma_theta + EPS
        
        kl_loss = ((e -output)).square().mean() + (sigma_tilde/sigma_theta).mean() - torch.log(sigma_tilde/sigma_theta).mean() 
        loss = kl_loss + loss1 + loss2  
        return loss


    def _process_val_batch(model,
                        cond_pred_model,
                        cond_pred_model_g,
                        batch_x, 
                        batch_y, 
                        batch_x_mark, 
                        batch_y_mark,
                        device):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(diffusion_steps + 1)] 
        y_se_by_batch_list = [[] for _ in range(diffusion_steps + 1)]
        minisample = 1
        
        batch_y_mark_input = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1) 

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), pred_len, dataset.num_features]
        ).to(device) 
        dec_inp_label = batch_x[:, -label_len :, :].to(device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq, pred_len = pred_len):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(b,
                                            # int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            minisample,
                                            (pred_len),
                                            dataset.num_features).cpu()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y.detach().cpu()
            else:
                gen_y_by_batch_list[current_t] = torch.concat([gen_y_by_batch_list[current_t], gen_y], dim=0).detach().cpu()
            return gen_y



        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=model.num_timesteps, size=(n // 2 + 1,)
        ).to(device)
        t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)[:n]
        
        y_0_hat_batch, _ = cond_pred_model(batch_x, batch_x_mark, dec_inp,batch_y_mark_input)
        gx = cond_pred_model_g(batch_x) + 1
        
        preds = []
        for i in range(100 //minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(device)
            y_T_mean_tile = y_0_hat_tile
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(device)

            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(device)

            future_mark_tile = batch_y_mark.repeat(repeat_n, 1, 1, 1)
            future_mark_tile = future_mark_tile.transpose(0, 1).flatten(0, 1).to(device)


            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(device)
            gen_y_box = []
            for _ in range(1):
                for _ in range(1):
                    y_tile_seq = p_sample_loop(model, x_tile, x_mark_tile, y_0_hat_tile, gx_tile, y_T_mean_tile,
                                                model.num_timesteps,
                                                model.alphas, model.one_minus_alphas_bar_sqrt,
                                                model.alphas_cumprod, model.alphas_cumprod_sum,
                                                model.alphas_cumprod_prev, model.alphas_cumprod_sum_prev,
                                                model.betas_tilde, model.betas_bar,
                                                model.betas_tilde_m_1, model.betas_bar_m_1,
                                                )
                gen_y = store_gen_y_at_step_t(config=None,
                                                config_diff=None,
                                                idx=model.num_timesteps, y_tile_seq=y_tile_seq)
                gen_y_box.append(gen_y.detach().cpu())
            outputs = torch.concat(gen_y_box, dim=1)

            f_dim =  0
            
            outputs = outputs[:, :, -pred_len:, f_dim:] # B, S, O, N

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred.detach().cpu()) # numberof_testbatch,  B, S, O, N
            # trues.append(true) # numberof_testbatch, B, T, N
            
        preds = torch.concat(preds, dim=1)
        batch_y = batch_y[:, -pred_len:, f_dim:].to(device) # B, T, N

        outs = preds.permute(0, 2, 3, 1) 
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (pred_len, dataset.num_features, 100)
        return outs, batch_y, y_0_hat_batch, gx


    from types import SimpleNamespace
    set_seed(1919 + seed_idx)
    args_nsdiff = SimpleNamespace(**args_dict)
    model = NsDiff(args_nsdiff, device).to(device)
    print('training the main model')
    cond_pred_model = ns_Transformer(args_nsdiff).float().to(device)
    cond_pred_model_g = SigmaEstimation(windows, 
                                        pred_len, 
                                        dataset.num_features, 
                                        512, 
                                        rolling_length).float().to(device)
    model_optim = Adam(
                [{'params': model.parameters()}, 
                {'params': cond_pred_model.parameters()}, 
                {'params': cond_pred_model_g.parameters()}], 
                lr=1e-4, 
                # weight_decay=self.l2_weight_decay
            )


    val_loss = []
    num_training_steps = 10

    best_val_loss = float('inf')
    # best_model_state = None

    best_model_state_diff = None
    best_model_state_fx =None
    best_model_state_gx =None

    best_step = -1

    import time 
    T1 = time.time()

    for step in range(num_training_steps):
        model.train()
        cond_pred_model.train()
        cond_pred_model_g.train()
        total_loss = 0
    ###################################################   train   #######################################
        for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader_cw.train_loader):


            model_optim.zero_grad()


            batch_history = batch_x.to(device).float()
            # batch_future = batch_y.to(device).float()
            batch_future= future_cw_train[i]['batch_future_cw'].to(device).float()
            batch_x_t_mark = batch_y_mark.to(device).float()
            batch_history_mark = batch_x_mark.to(device).float()

            loss = _process_train_batch(model = model,
                            cond_pred_model = cond_pred_model,
                            cond_pred_model_g = cond_pred_model_g,
                            batch_x = batch_history, 
                            batch_y = batch_future, 
                            batch_x_mark = batch_history_mark, 
                            batch_y_mark = batch_x_t_mark,
                            device = device)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), 1.)
            # torch.nn.utils.clip_grad_norm_(
            #     cond_pred_model.parameters(), 1.)
            # torch.nn.utils.clip_grad_norm_(
            #     cond_pred_model_g.parameters(), 1.)
            
            model_optim.step()


            total_loss += loss.item()
        total_loss = total_loss / len(dataloader_cw.train_loader)
        

    ###################################################   val   #######################################
        with torch.no_grad():
            model.eval()
            cond_pred_model.eval()
            cond_pred_model_g.eval()
            val_total = 0
            for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader_cw.val_loader):

                batch_history = batch_x.to(device).float()
                # batch_future = batch_y.to(device).float()
                batch_future= future_cw_val[i]['batch_future_cw'].to(device).float()
                batch_x_t_mark = batch_y_mark.to(device).float()
                batch_history_mark = batch_x_mark.to(device).float()

                loss = _process_train_batch(model = model,
                                            cond_pred_model = cond_pred_model,
                                            cond_pred_model_g = cond_pred_model_g,
                                            batch_x = batch_history, 
                                            batch_y = batch_future, 
                                            batch_x_mark = batch_history_mark, 
                                            batch_y_mark = batch_x_t_mark,
                                            device = device)
                val_total += loss.item()
            val_avg = val_total / len(dataloader_cw.val_loader)
            val_loss.append(val_avg)

    ###################################################   test   #######################################
            # visualize the last batch
            test_total = 0
            test_total_mean = 0
            test_total_sigma = 0
            for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader_cw.test_loader):
                batch_history = batch_x.to(device).float()
                # batch_future = batch_y.to(device).float()
                batch_future= future_cw_test[i]['batch_future_cw'].to(device).float()
                batch_x_t_mark = batch_y_mark.to(device).float()
                batch_history_mark = batch_x_mark.to(device).float()
                
                test_loss = _process_train_batch(model = model,
                                                cond_pred_model = cond_pred_model,
                                                cond_pred_model_g = cond_pred_model_g,
                                                batch_x = batch_history, 
                                                batch_y = batch_future, 
                                                batch_x_mark = batch_history_mark, 
                                                batch_y_mark = batch_x_t_mark,
                                                device = device)

                test_total += test_loss.item()
                test_avg = test_total / len(dataloader_cw.test_loader)
            
            

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_step = step
                best_model_state_diff = {k: v.clone() for k, v in model.state_dict().items()}
                best_model_state_fx = {k: v.clone() for k, v in cond_pred_model.state_dict().items()}
                best_model_state_gx = {k: v.clone() for k, v in cond_pred_model_g.state_dict().items()}


        # print(f"Step {step}: Train Loss = {total_loss:.4f} | Val Loss = {val_avg:.4f} | Test Loss = {test_avg:.4f}")


    T2 = time.time()
    if best_model_state_diff is not None:
        model.load_state_dict(best_model_state_diff)
        cond_pred_model.load_state_dict(best_model_state_fx)
        cond_pred_model_g.load_state_dict(best_model_state_gx)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")


    import matplotlib.pyplot as plt
    import torch
    set_seed(1920 + seed_idx)
    model.eval()
    cond_pred_model.eval()
    cond_pred_model_g.eval()


    origin_y = origin_y.to(device).float()
    batch_x = batch_x.to(device).float()
    batch_x = future_cw_test[0]['batch_history'].to(device).float()
    batch_y = future_cw_test[0]['batch_future_cw'].to(device).float() ################# change here!
    batch_x_date_enc = future_cw_test[0]['batch_history_mark'].to(device).float()
    batch_y_date_enc = future_cw_test[0]['batch_future_mark'].to(device).float()
    print('calculating metrics')
    with torch.no_grad():
        # batch_x = batch_x.to(exp.device).float()
        # batch_x_mark = batch_x_date_enc.to(exp.device).float()
        # batch_y = batch_y.to(exp.device).float()
        # batch_y_mark = batch_y_date_enc.to(exp.device).float()
        origin_y = origin_y.to(device).float()
        batch_x = batch_x.to(device).float()
        batch_x = future_cw_test[0]['batch_history'].to(device).float()
        batch_y_cw = future_cw_test[0]['batch_future_cw'].to(device).float() ################# change here!
        batch_y = future_cw_test[0]['batch_future'].to(device).float() ################# change here!
        batch_x_mark = future_cw_test[0]['batch_history_mark'].to(device).float()
        batch_y_mark = future_cw_test[0]['batch_future_mark'].to(device).float()
        preds, truths, y_0_hat_batch, gx = _process_val_batch(model = model,
                                        cond_pred_model = cond_pred_model,
                                        cond_pred_model_g = cond_pred_model_g,
                                        batch_x = batch_x, 
                                        batch_y = batch_y, 
                                        batch_x_mark = batch_x_mark, 
                                        batch_y_mark = batch_y_mark,
                                        device = device)

    
    generated_batch = []
    cov_sqrt_test = future_cw_test[0]['cov_sqrt'].to(device)
    miu_test = future_cw_test[0]['miu_pred'].to(device)
    for iii in range(preds.shape[-1]):
        pred_temp = torch.einsum('btij,btj->bti', cov_sqrt_test.to(device), preds[:,:,:,iii].to(device)) + miu_test.to(device)
        # pred_temp = preds[:,:,:,iii]
        generated_batch.append(pred_temp.unsqueeze(-1))




    metrics = MetricCollection(metrics={"crps": CRPS(),
                                        "crps_sum": CRPSSum(),
                                        "qice": QICE(),
                                        "picp": PICP(),
                                        "mse": ProbMSE(),
                                        "mae":ProbMAE(),
                                        "rmse": ProbRMSE(),
                                        }
                                        )
    metrics.to("cpu")


    metrics.reset()
    metrics.update(torch.cat(generated_batch,dim=-1).float().detach().cpu(), batch_y.detach().cpu())


    metrics_dict = {name: float(metric.compute()) for name, metric in metrics.items()}


    pred_xy = [torch.cat([batch_x,x.squeeze(-1)],dim=1) for x in generated_batch]
    pred_xy = torch.cat(pred_xy,dim=0)

    context_fid = Context_FID(ori_data = torch.cat([batch_x,batch_y],dim=1).detach().cpu().numpy(), 
                generated_data = pred_xy.detach().cpu().numpy())
    cacf_list = []
    for iii in range(len(generated_batch)):
        cacf_list.append((cacf_torch(x = torch.cat([batch_x,batch_y],dim=1)) \
                        -cacf_torch(x =  torch.cat([batch_x,generated_batch[iii].squeeze(-1)],dim=1))).abs().mean())
    cacf_mean , cacf_std = torch.tensor(cacf_list).mean(),torch.tensor(cacf_list).std()

    with open(log_file, "a", encoding="utf-8") as f:
        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed_idx": seed_idx,
            "metrics": metrics_dict,
            "Context_FID": float(context_fid),
            "cacf_mean": float(cacf_mean),
            "cacf_std": float(cacf_std)
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

