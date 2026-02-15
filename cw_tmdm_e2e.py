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



    device = "cuda:1"
    # seed_idx = 0
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
    assert os.path.exists(data_path), f"didnt found: {data_path}"

    df = pd.read_csv(data_path)
    # display(df.head())


    config = {
        "teacher_force": False,
        "mixup": False,

        "matrix_norm_weight": [(7*192)**0.5 * 1e-1, 1, 0.],
        "fft_weight": [1, 0.],
        "eign_penalty": 50,
        "eps_eign_min": 1e-1,
        "penalty_method": "hard",
        'num_training_steps': 20,

        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 3,
        "dropout": 0.1,

        "p_hidden_layers": 2,
        "p_hidden_dims": [128, 128],

        "windows": 168,
        "horizon": 1,
        "pred_len": 192,
        "label_len": 168 // 2,
        'num_features': 7,

        "batch_size": 64,
        "num_worker": 0,
        'dataset_type': "ETTh1",
        'data_path': "ts_datasets",
        'scaler_type': "StandardScaler",

        'lr': 1e-4,
        'weight_decay': 5e-4,

        'window_size': 95,
        'pad_mode': 'reflect',
    }

    lambda_min = config['eps_eign_min']
    weigen = config['eign_penalty']
    log_file = f"cw_tmdm_lambda_{lambda_min}_weigen{weigen}_e2e.txt"


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

        def unpack_cholesky_upper(self, flat_triu):
            B, T, _ = flat_triu.shape
            D = self.num_feature
            U = torch.zeros(B, T, D, D, device=flat_triu.device)
            triu_idx = torch.triu_indices(D, D, device=flat_triu.device)
            U[:, :, triu_idx[0], triu_idx[1]] = flat_triu
            U[:, :, range(D), range(D)] = F.softplus(U[:, :, range(D), range(D)])  # if you want, you can add a small positive number at here. but the numerical stability seems good, so we didn't add
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
            dec_out[:,:,:num_features] = dec_out[:,:,:num_features] * std_enc[:,:,:num_features] + mean_enc[:,:,:num_features]

            
            miu_pred = dec_out[:,:,:num_features]
            U_flat = dec_out[:,:,num_features:]

            
            U = self.unpack_cholesky_upper(U_flat)   
            Sigma_pred = U.transpose(-1, -2) @ U

            return miu_pred[:, -self.pred_len:, :], Sigma_pred[:, -self.pred_len:, :, :], U[:, -self.pred_len:, :, :]




    def compute_batch_xxT(batch_x):
        B, T, D = batch_x.shape
        device = batch_x.device
        batch_xxT = batch_x.unsqueeze(-1) @ batch_x.unsqueeze(-2) # [B, T, D, D]

        triu_indices = torch.triu_indices(D, D)
        # Result shape [B, T, D*(D+1)//2]
        batch_upper_triangular = batch_xxT[:, :, triu_indices[0], triu_indices[1]]
        return batch_xxT, batch_upper_triangular


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


    def soft_hinge_all_mean(eigvals, eps=1e-3, beta=5.0): # never use it
        eps_t = torch.as_tensor(eps, dtype=eigvals.dtype, device=eigvals.device)
        gaps  = eps_t - eigvals                 # [B,T,D]
        softg = F.softplus(beta * gaps) / beta  
        return softg.mean()                     


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
        loss_miu = F.mse_loss(miu_pred, theta_true)
        fft_loss_miu = (torch.fft.rfft(miu_pred, dim=1) - torch.fft.rfft(theta_true, dim=1)).abs().mean()  # dont use it

        diff = (sigma_pred - theta_outer_true) / batch_xxT_std  # actually we never normalize it. batch_xxT_std is always 1
        loss_fro = diff.pow(2).mean()
        svals = torch.linalg.svdvals(diff)  # [B, T, D]
        loss_svd = svals.mean()
        

        if eign_penalty > 0:
            cov_consistency = sigma_pred
            eigvals = torch.linalg.eigvalsh(cov_consistency)
            if penalty_method == 'hard':
                posdef_penalty = torch.relu(eps_eign_min - eigvals).mean() # the penalty
            elif penalty_method == 'soft':
                posdef_penalty = soft_hinge_all_mean(eigvals, eps=eps_eign_min, beta=20.0)
        else:
            posdef_penalty = 0
        
        if verbose:
            if eign_penalty > 0:
                print(f'l2 loss:{loss_miu.item()}, f norm loss:{loss_fro.item()}, svd norm loss:{loss_svd.item()}, penalty:{posdef_penalty.item()}')
            else:
                print(f'l2 loss:{loss_miu.item()}, f norm loss:{loss_fro.item()}, svd norm loss:{loss_svd.item()}, penalty: not used')
        fft_loss_cov = (torch.fft.rfft(diff, dim=1) ).abs().mean()  # dont use it
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

                inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals_clamped)) @ eigvecs.T
                inv_sqrt_all[b, t] = inv_sqrt
                whitened[b, t] = inv_sqrt @ residual[b, t]

                sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                sqrt_all[b, t] = sqrt

        return whitened, inv_sqrt_all, sqrt_all


    def load_saved_model(model_conditional_mean, seed_idx, model_path_dir='results/runs/cond_mean_cov_model'):
        os.makedirs(model_path_dir, exist_ok=True)
        model_path = os.path.join(model_path_dir, f'model_cond_mean_cov_{seed_idx}.pth')

        if os.path.exists(model_path):
            print(f"Found model at {model_path}, loading weights...")
            model_conditional_mean.load_state_dict(torch.load(model_path))
        # else:
        #     print(f"No existing model at {model_path}, saving new model...")
        #     torch.save(model_conditional_mean.state_dict(), model_path)


    set_seed(514 + seed_idx)
    model_conditional_mean = ns_Transformer().float().to(device)
    print(sum(p.numel() for p in model_conditional_mean.parameters()))



    import torch

    def stabilize_upper(U, eps_diag):
        diag = U.diagonal(dim1=-2, dim2=-1)         # [B,T,D]
        diag_new = diag.abs().clamp_min(eps_diag)   # Apply lower bound and prevent negative sign
        U_stable = U.clone()
        U_stable.diagonal(dim1=-2, dim2=-1).copy_(diag_new)
        return U_stable


    def whiten_sequence_fast(theta, conditional_mean, upper_mat,
                            eps=0.1, verbose=False):
        """
        theta: [B, T, D]
        conditional_mean: [B, T, D]
        upper_mat: [B, T, D, D], Upper-triangular U such that Sigma ≈ U^T U

        Returns:
            whitened: [B, T, D]
            inv_sqrt_all: [B, T, D, D]
            sqrt_all: [B, T, D, D]
        """

        residual = theta - conditional_mean      # [B, T, D]
        B, T, D = residual.shape

        # --- Stabilize U (clip diagonal only)---
        # eps_diag = eps_eign_min**0.5
        # U_stable = stabilize_upper(upper_mat, eps_diag)
        U_stable = upper_mat

        if verbose:
            orig_diag = upper_mat.diagonal(dim1=-2, dim2=-1)
            new_diag  = U_stable.diagonal(dim1=-2, dim2=-1)
            num_clipped = (new_diag > orig_diag.abs()).sum().item()
            print(f"[whiten_sequence_fast] diag entries clipped: {num_clipped}")

        # --- Key fix: use U^T for triangular solve, compatible with older PyTorch versions ---
        whitened = torch.linalg.solve_triangular(
            U_stable.transpose(-1, -2),    # U^T is lower triangular
            residual.unsqueeze(-1),        # [B,T,D,1]
            left=True,
            upper=False                    # lower triangular
        ).squeeze(-1)                      # [B,T,D]

        # --- Lower-triangular square root L = U^T ---
        sqrt_all = U_stable.transpose(-1, -2).contiguous()

        # --- Explicitly compute L^{-1} (if needed)---
        I = torch.eye(D, dtype=theta.dtype, device=theta.device).expand(B, T, D, D)
        inv_sqrt_all = torch.linalg.solve_triangular(
            sqrt_all,                      # lower-triangular
            I,
            left=True,
            upper=False
        )

        return whitened, inv_sqrt_all, sqrt_all



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
        model = TS2Vec(input_dims=ori_data.shape[-1], device=device, batch_size=512, lr=0.001, output_dims=320,
                    max_train_length=3000)
        model.fit(ori_data, verbose=False)
        ori_represenation = model.encode(ori_data, encoding_window='full_series')
        gen_represenation = model.encode(generated_data, encoding_window='full_series')
        idx = np.random.permutation(ori_data.shape[0])
        ori_represenation = ori_represenation[idx]
        gen_represenation = gen_represenation[idx]
        results = calculate_fid(ori_represenation, gen_represenation)
        return results


    EPS = 10e-8
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
        # NsDiff: y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + noise
        y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + sqrt_one_minus_alpha_bar_t * noise
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



    def p_sample(model, x, x_mark, y, y_0_hat, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt):
        """
        Reverse diffusion process sampling -- one time step.

        y: sampled y at time step t, y_t.
        y_0_hat: prediction of pre-trained guidance model.
        y_T_mean: mean of prior distribution at timestep T.
        We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
            guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
            in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
        """
        device = next(model.parameters()).device
        z = torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
        t = torch.tensor([t]).to(device)
        alpha_t = extract(alphas, t, y)
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
        # y_t_m_1 posterior mean component coefficients
        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
            sqrt_one_minus_alpha_bar_t.square())
        eps_theta = model(x, x_mark, 0, y, y_0_hat, t).to(device).detach()
        # y_0 reparameterization
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
        # posterior mean
        y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
        # posterior variance
        beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
        y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
        return y_t_m_1

    def p_sample_t_1to0(model, x, x_mark, y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt):
        device = next(model.parameters()).device
        t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        eps_theta = model(x, x_mark, 0, y, y_0_hat, t).to(device).detach()
        # y_0 reparameterization
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
        y_t_m_1 = y_0_reparam.to(device)
        return y_t_m_1


    def p_sample_loop(model, x, x_mark, y_0_hat, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt):
        device = next(model.parameters()).device
        z = torch.randn_like(y_T_mean).to(device)
        cur_y = z + y_T_mean  # sample y_T
        y_p_seq = [cur_y]
        for t in reversed(range(1, n_steps)):  # t from T to 2
            y_t = cur_y
            cur_y = p_sample(model, x, x_mark, y_t, y_0_hat, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
            y_p_seq.append(cur_y)
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(model, x, x_mark, y_p_seq[-1], y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
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
            # x:     B x S x E
            # stats: B x 1 x E
            # y:     B x O
            batch_size = x.shape[0]
            x = self.series_conv(x)  # B x 1 x E
            x = torch.cat([x, stats], dim=1)  # B x 2 x E
            x = x.view(batch_size, -1)  # B x 2E
            y = self.backbone(x)  # B x O

            return y


    class Model(nn.Module):
        """
        Non-stationary Transformer
        """

        def __init__(self, configs):
            super(Model, self).__init__()
            self.pred_len = configs.pred_len
            self.seq_len = configs.seq_len
            self.label_len = configs.label_len
            self.output_attention = configs.output_attention

            # Embedding
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            # Encoder
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
            # Decoder
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

            # Model Inference
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

            mean = self.z_mean(enc_out)
            logvar = self.z_logvar(enc_out)

            z_sample = self.reparameterize(mean, logvar)

            # dec_out = self.z_out(torch.cat([z_sample, dec_out], dim=-1))
            enc_out = self.z_out(z_sample)

            KL_z = self.KL_loss_normal(mean, logvar)

            dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

            # De-normalization
            # dec_out = dec_out * std_enc + mean_enc ###################################### please must # it

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :], dec_out, KL_z, z_sample  # [B, L, D]



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
        def __init__(self, config, MTS_args):
            super(ConditionalGuidedModel, self).__init__()
            n_steps = config.diffusion.timesteps + 1
            self.cat_x = config.model.cat_x 
            self.cat_y_pred = config.model.cat_y_pred 
            
            data_dim = MTS_args.enc_in * 2 if self.cat_y_pred else MTS_args.enc_in

            self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
            self.lin2 = ConditionalLinear(128, 128, n_steps)
            self.lin3 = ConditionalLinear(128, 128, n_steps)
            self.lin4 = nn.Linear(128, MTS_args.enc_in)

        def forward(self, x, y_t, y_0_hat, t):
            # x/y_t/y_0_hat: (B,T,N)
            # t:(B,)
            if self.cat_x:
                if self.cat_y_pred:
                    eps_pred = torch.cat((y_t, y_0_hat), dim=-1) 
                else:
                    eps_pred = torch.cat((y_t, x), dim=2) 
            else:
                if self.cat_y_pred:
                    eps_pred = torch.cat((y_t, y_0_hat), dim=2)
                else:
                    eps_pred = y_t
            if y_t.device.type == 'mps':
                eps_pred = self.lin1(eps_pred, t)
                eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

                eps_pred = self.lin2(eps_pred, t)
                eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

                eps_pred = self.lin3(eps_pred, t)
                eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            else:
                eps_pred = F.softplus(self.lin1(eps_pred, t))
                eps_pred = F.softplus(self.lin2(eps_pred, t))
                eps_pred = F.softplus(self.lin3(eps_pred, t))
            eps_pred = self.lin4(eps_pred)
            return eps_pred


    class DeterministicFeedForwardNeuralNetwork(nn.Module):

        def __init__(self, dim_in, dim_out, hid_layers,
                    use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
            super(DeterministicFeedForwardNeuralNetwork, self).__init__()
            self.dim_in = dim_in  # dimension of nn input
            self.dim_out = dim_out  # dimension of nn output
            self.hid_layers = hid_layers  # nn hidden layer architecture
            self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
            self.use_batchnorm = use_batchnorm  # whether apply batch norm
            self.negative_slope = negative_slope  # negative slope for LeakyReLU
            self.dropout_rate = dropout_rate
            layers = self.create_nn_layers()
            self.network = nn.Sequential(*layers)

        def create_nn_layers(self):
            layers = []
            for idx in range(len(self.nn_layers) - 1):
                layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
                if self.use_batchnorm:
                    layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
                layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
                layers.append(nn.Dropout(p=self.dropout_rate))
            layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
            return layers

        def forward(self, x):
            return self.network(x)





    class TMDM(nn.Module):
        """
        Vanilla Transformer
        """

        def __init__(self, configs, device):
            super(TMDM, self).__init__()

            with open(configs.diffusion_config_dir, "r") as f:
                config = yaml.unsafe_load(f)
                diffusion_config = dict2namespace(config)

            diffusion_config.diffusion.timesteps = configs.timesteps
            
            self.args = configs
            self.device = device
            self.diffusion_config = diffusion_config

            self.model_var_type = diffusion_config.model.var_type
            self.num_timesteps = diffusion_config.diffusion.timesteps
            self.vis_step = diffusion_config.diffusion.vis_step
            self.num_figs = diffusion_config.diffusion.num_figs
            self.dataset_object = None

            betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                    start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
            betas = self.betas = betas.float().to(self.device)
            self.betas_sqrt = torch.sqrt(betas)
            alphas = 1.0 - betas
            self.alphas = alphas
            self.one_minus_betas_sqrt = torch.sqrt(alphas)
            alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
            self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
            self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
            if diffusion_config.diffusion.beta_schedule == "cosine":
                self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            alphas_cumprod_prev = torch.cat(
                [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
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
            if self.model_var_type == "fixedlarge":
                self.logvar = betas.log()
                # torch.cat(
                # [posterior_variance[1:2], betas[1:]], dim=0).log()
            elif self.model_var_type == "fixedsmall":
                self.logvar = posterior_variance.clamp(min=1e-20).log()

            self.tau = None  # precision fo test NLL computation

            # CATE MLP
            self.diffussion_model = ConditionalGuidedModel(diffusion_config, self.args)

            self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                            configs.dropout)


        def forward(self, x, x_mark, y, y_t, y_0_hat, t):
            enc_out = self.enc_embedding(x, x_mark) #  B, T, d_model
            dec_out = self.diffussion_model(enc_out, y_t, y_0_hat, t)

            return dec_out





    from dataclasses import dataclass, field
    import sys
    from typing import List, Dict
    import os
    import wandb
    import torch
    from dataclasses import dataclass, asdict, field
    from torch_timeseries.nn.embedding import freq_map
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

    import numpy as np
    import torch.distributed as dist
    import torch
    from tqdm import tqdm
    import concurrent.futures
    from types import SimpleNamespace




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
        return 0.5 * torch.mean(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


    beta_start: float =  0.0001
    beta_end: float =  0.5
    d_model: int =  512
    n_heads: int =  8
    e_layers: int =  2
    d_layers: int =  1
    d_ff: int =  1024
    diffusion_steps :int = 100
    moving_avg: int =  25
    factor: int =  3
    distil: bool =  True
    dropout: float =  0.05
    activation: str = 'gelu'
    k_z: float =  1e-2
    k_cond: int =  1
    d_z: int =  8
    CART_input_x_embed_dim : int= 32
    p_hidden_layers : int = 2


    data_path = "ts_datasets"
    dataset_type = "ETTh1"
    windows = 168         # Input window length
    pred_len = 192         # Prediction length
    batch_size = 32
    epochs = 10           # Adjust as needed
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
    label_len = windows // 2
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
        "diffusion_config_dir" : "NsDiff-main/configs/tmdm.yml",
    }

    args = SimpleNamespace(**args_dict)



    def _process_train_batch(model, 
                            cond_pred_model,
                            batch_x, batch_y, batch_x_mark, batch_y_mark, device = device):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        # Time Diff need the batch_x to be a even number
        
        
        # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        # dec_inp = torch.cat([batch_x[:, -label_len:, :], dec_inp], dim=1).float().to(device)

        batch_y = torch.concat([batch_x[:, -label_len:, :], batch_y], dim=1)
        batch_y_mark = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1)

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
        _, y_0_hat_batch, KL_loss, z_sample = cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

        loss_vae_all = loss_vae + k_z * KL_loss
        # y_0_hat_batch = z_sample

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(device)

        y_t_batch = q_sample(batch_y, y_T_mean, model.alphas_bar_sqrt,
                                model.one_minus_alphas_bar_sqrt, t, noise=e)

        output = model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)
        # loss = (e[:, -args.pred_len:, :] - output[:, -args.pred_len:, :]).square().mean()
        loss = (e - output).square().mean() + args.k_cond*loss_vae_all
        return loss


    def _process_val_batch(model, 
                        cond_pred_model,
                        batch_x, batch_y, batch_x_mark, batch_y_mark, device = device):
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
        minisample = 10
        
        batch_y = torch.concat([batch_x[:, -label_len:, :], batch_y], dim=1)
        batch_y_mark = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), pred_len, dataset.num_features]
        ).to(device)
        dec_inp_label = batch_x[:, -label_len :, :].to(device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)


        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(b,
                                            # int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            minisample,
                                            (config.label_len + config.pred_len),
                                            config.c_out).cpu()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = torch.concat([gen_y_by_batch_list[current_t], gen_y], dim=0)
            return gen_y


        # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        # dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=model.num_timesteps, size=(n // 2 + 1,)
        ).to(device)
        t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)[:n]
        
        _, y_0_hat_batch, _, z_sample = cond_pred_model(batch_x, batch_x_mark, dec_inp,batch_y_mark)
        preds = []
        for i in range(model.diffusion_config.testing.n_z_samples //minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(device)
            y_T_mean_tile = y_0_hat_tile
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(device)

            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(device)

            gen_y_box = []
            for _ in range(model.diffusion_config.testing.n_z_samples_depart):
                for _ in range(model.diffusion_config.testing.n_z_samples_depart):
                    y_tile_seq = p_sample_loop(model, x_tile, x_mark_tile, y_0_hat_tile, y_T_mean_tile,
                                                model.num_timesteps,
                                                model.alphas, model.one_minus_alphas_bar_sqrt)
                gen_y = store_gen_y_at_step_t(config=model.args,
                                                config_diff=model.diffusion_config,
                                                idx=model.num_timesteps, y_tile_seq=y_tile_seq)
                gen_y_box.append(gen_y)
            outputs = torch.concat(gen_y_box, dim=1)

            f_dim = -1 if args.features == 'MS' else 0
            
            outputs = outputs[:, :, -pred_len:, f_dim:] # B, S, O, N

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred) # numberof_testbatch,  B, S, O, N
            # trues.append(true) # numberof_testbatch, B, T, N
        preds = torch.concat(preds, dim=1)
        batch_y = batch_y[:, -pred_len:, f_dim:].to(device) # B, T, N

        outs = preds.permute(0, 2, 3, 1)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (pred_len, dataset.num_features, model.diffusion_config.testing.n_z_samples)
        return outs, batch_y


    model = TMDM(args, device).to(device)
    print(sum(p.numel() for p in model.parameters()))
    cond_pred_model = Model(args).float().to(device)


    optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(),
                "lr": config['lr'],
                "weight_decay": 5e-4},

                {"params": cond_pred_model.parameters(),
                "lr": config['lr'],
                "weight_decay": 5e-4},
                
                {"params": model_conditional_mean.parameters(),
                "lr": config['lr'],
                "weight_decay": 5e-4}
            ]
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

            # we nerver use it ↓
            # ############# normalize the sigma, this is to keep the scale of cov-loss ###########
            # batch_xxT_trig = batch_xxT_trig / train_set_xxT_up_trig_std
            # batch_yyT_trig = batch_yyT_trig / train_set_xxT_up_trig_std
            # batch_xxT = batch_xxT / train_set_xxT_std
            # batch_yyT = batch_yyT / train_set_xxT_std
            # ############# normalize the sigma ############################

            # === mixup === # it is inspired by timediff. but we also never use it
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

            if teacher_force:
                future_mixup_weight = torch.rand((batch_y.shape[0],1,1)).to(device)
                miu_pred, sigma_pred, U = model_conditional_mean(x_enc = batch_x, 
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
                miu_pred, sigma_pred, U = model_conditional_mean(x_enc = batch_x, 
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
            future_cw, cov_sqrt_inv, cov_sqrt = whiten_sequence_fast(batch_y, 
                                                conditional_mean=miu_pred, 
                                                upper_mat=U, eps=0, verbose = False)
                

            loss_tmdm = _process_train_batch(model = model,
                            cond_pred_model = cond_pred_model,
                            batch_x = batch_x, 
                            batch_y = future_cw, 
                            batch_x_mark = batch_x_mark, 
                            batch_y_mark = batch_y_mark,
                            device = device)


            loss_jmce = joint_loss_fn(theta_true=batch_y, 
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
            loss = (loss_tmdm + loss_jmce) / 2

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model_conditional_mean.parameters(), 1.)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.)
            torch.nn.utils.clip_grad_norm_(
                cond_pred_model.parameters(), 1.)
            
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
                
                miu_pred, sigma_pred, U = model_conditional_mean(x_enc = batch_x, 
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
                    

                future_cw, cov_sqrt_inv, cov_sqrt = whiten_sequence_fast(batch_y, 
                                                    conditional_mean=miu_pred, 
                                                    upper_mat=U, eps=0, verbose = False)
                    

                loss_tmdm = _process_train_batch(model = model,
                                cond_pred_model = cond_pred_model,
                                batch_x = batch_x, 
                                batch_y = future_cw, 
                                batch_x_mark = batch_x_mark, 
                                batch_y_mark = batch_y_mark,
                                device = device)


                loss_jmce = joint_loss_fn(theta_true=batch_y, 
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
                loss = (loss_tmdm + loss_jmce) / 2
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
                
                miu_pred, sigma_pred, U = model_conditional_mean(x_enc = batch_x, 
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
                
                future_cw, cov_sqrt_inv, cov_sqrt = whiten_sequence_fast(batch_y, 
                                                    conditional_mean=miu_pred, 
                                                    upper_mat=U, eps=0, verbose = False)
                    

                loss_tmdm = _process_train_batch(model = model,
                                cond_pred_model = cond_pred_model,
                                batch_x = batch_x, 
                                batch_y = future_cw, 
                                batch_x_mark = batch_x_mark, 
                                batch_y_mark = batch_y_mark,
                                device = device)


                loss_jmce = joint_loss_fn(theta_true=batch_y, 
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
                test_loss = (loss_tmdm + loss_jmce) / 2
                    
                
                # #################### de normalize, then we use the predicted cov, whiten the sequence, visualize ####
                # sigma_pred = sigma_pred * train_set_xxT_std
                # batch_yyT = batch_yyT * train_set_xxT_std
                # batch_yyT_trig = batch_yyT_trig * train_set_xxT_up_trig_std
                # #################### de normalize ####################

                
                # batch_y_corr_score, batch_y_cw_corr_score = compute_corr_score(batch_y,y_test_cw)

                whiten_score_y, _ = average_r2_correlation_metric(batch_y, normalize = True, method='linear')
                whiten_score_y_cw, _ = average_r2_correlation_metric(future_cw, normalize = True, method='linear')
                whiten_score_y_centralized, _ = average_r2_correlation_metric(batch_y-miu_pred, normalize = True, method='linear')

                
                test_total += test_loss.item()
                test_total_mean += loss_func(batch_y, miu_pred).item()
                test_total_sigma += loss_func(batch_yyT, sigma_pred).item()
            test_avg = test_total / len(dataloader.test_loader)
            test_mean_avg = test_total_mean / len(dataloader.test_loader)
            test_sigma_avg = test_total_sigma / len(dataloader.test_loader)

            triu_indices = torch.triu_indices(num_features, num_features)

            if val_avg < best_val_loss and step :
                best_val_loss = val_avg
                best_step = step
                best_model_state_jmce = {k: v.clone() for k, v in model_conditional_mean.state_dict().items()}
                best_model_state_tmdm = {k: v.clone() for k, v in model.state_dict().items()}
                best_model_state_tmdm_cond_mean = {k: v.clone() for k, v in cond_pred_model.state_dict().items()}
            


        print(f"{step}:Train={total_loss:.4f}|Val={val_avg:.4f}|Test={test_avg:.4f}|whiten_score_y={whiten_score_y:.4f}|whiten_score_y_cw={whiten_score_y_cw:.4f}|whiten_score_y_cent={whiten_score_y_centralized:.4f}")




    if best_model_state_jmce is not None:
        model_conditional_mean.load_state_dict(best_model_state_jmce)
        model.load_state_dict(best_model_state_tmdm)
        cond_pred_model.load_state_dict(best_model_state_tmdm_cond_mean)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")



    future_cw_test = {}
    # with torch.inference_mode():
    model_conditional_mean.eval()
    for i, (batch_history,
            batch_future,
            origin_history,
            origin_future,
            batch_history_mark,
            batch_future_mark,
            ) in enumerate(dataloader.test_loader):
        print(f'{i}-th batch is whitened')
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

        # print(batch_history.shape, batch_his_xxT_trig.shape, batch_history_mark.shape, dec_inp.shape, batch_x_mark.shape, )
        miu_pred, sigma_pred, U = model_conditional_mean(
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
                                    upper_mat=U.detach(), eps=eps_eign_min, verbose = False)
        future_cw_test[i] = {'batch_history': batch_history.detach(),
                                'batch_future': batch_future.detach(),
                                'batch_history_mark': batch_history_mark.detach(),
                                'batch_future_mark': batch_future_mark.detach(),
                                'batch_future_cw': future_cw.detach(),
                                'batch_cov_sqrt_inv': cov_sqrt_inv.detach(),
                                'cov_sqrt': cov_sqrt.detach(),
                                'miu_pred': miu_pred.detach(),
                                }


    import matplotlib.pyplot as plt
    import torch
    set_seed(1920 + seed_idx)
    model.eval()
    cond_pred_model.eval()


    origin_y = origin_y.to(device).float()
    batch_x = batch_x.to(device).float()
    batch_x = future_cw_test[0]['batch_history'].to(device).float()
    batch_y = future_cw_test[0]['batch_future_cw'].to(device).float() ################# change here!
    batch_x_date_enc = future_cw_test[0]['batch_history_mark'].to(device).float()
    batch_y_date_enc = future_cw_test[0]['batch_future_mark'].to(device).float()

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
        preds, truths = _process_val_batch(model,cond_pred_model,batch_x, batch_y, batch_x_mark, batch_y_mark)



    generated_batch = []
    cov_sqrt_test = future_cw_test[0]['cov_sqrt'].to(device)
    miu_test = future_cw_test[0]['miu_pred'].to(device)
    for iii in range(preds.shape[-1]):
        pred_temp = torch.einsum('btij,btj->bti', cov_sqrt_test.to(device), preds[:,:,:,iii].to(device)) + miu_test.to(device)
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


