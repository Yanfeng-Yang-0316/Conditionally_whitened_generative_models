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



    device = "cuda:2"
    # seed_idx = 1
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
    assert os.path.exists(data_path), f"cannot find {data_path}"

    df = pd.read_csv(data_path)

    num_features = df.shape[1] - 1

    # display(df.head())


    train_set = torch.tensor(df.iloc[:int(0.7*df.shape[0]),1:].values).float() # make sure we only use the train set
    # dont forget normalize the train set
    train_set_normalized = (train_set - train_set.mean(dim=0)) / train_set.std(dim=0)

    train_set_xxT = train_set_normalized.unsqueeze(-1) @ train_set_normalized.unsqueeze(-2)
    train_set_xxT_std = train_set_xxT.std(dim=0).unsqueeze(0).unsqueeze(0)
    triu_indices = torch.triu_indices(num_features, num_features)
    train_set_xxT_up_trig_std = train_set_xxT_std[:, :, triu_indices[0], triu_indices[1]]

    train_set_xxT_std = train_set_xxT_std.to(device)
    train_set_xxT_up_trig_std = train_set_xxT_up_trig_std.to(device)


    # train_set_xxT_std.shape,train_set_xxT_up_trig_std.shape


    config = {
        # always false
        "teacher_force": False,
        "mixup": False,

        # loss weight 
        "matrix_norm_weight": [(7*192)**0.5 * 0.05, 1, 0.], # [f norm, svd norm, fft norm] fft norm is always 0
        "fft_weight": [1, 0.], # [L2 norm, fft norm] fft norm is always 0
        "eign_penalty": 50, # w_Eigen
        "eps_eign_min": 0.05, # lambda_min
        "penalty_method": "hard", # use relu
        'num_training_steps': 20,

        # Transformer 
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

        # ts 
        "windows": 168,
        "horizon": 1,
        "pred_len": 192,
        "label_len": 168 // 2,

        # DataLoader 
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

    config_diffusion = {
        
        # diffusion 
        'diff_total_steps': 1000,
        'noise_schedule': 'linear',
        'noise_start': 1e-4,
        'noise_end': 2e-2,
        
        # Transformer 
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 3,
        "d_layers": 3,
        "mlp_hidden_times": 4,
        "dropout": 0.1,

        "windows": 168,
        "horizon": 1,
        "pred_len": 192,
        "label_len": 168 // 2,

        # DataLoader 
        "batch_size": 128,
        "num_worker": 0,
        'dataset_type': "ETTh1",
        'data_path': "ts_datasets_flow_cw",
        'scaler_type': "StandardScaler",
        
        'num_training_steps': 20,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'use_fft': 192 ** 0.5 / 5,
        'ot_beta': -1,          # ot_beta=-1: not use ot (ot_beta should >=0). when ot_beta = 0, it is original ot. when # ot_beta>0, it is conditional ot
        'mask_cond_prob': 0.75,
    }
    # please always use ot_beta=-1





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
        fast_val = True,
        fast_test = True,
    )


    dataloader.train_loader.__len__(),dataloader.val_loader.__len__(),dataloader.test_loader.__len__()


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

            if self.output_attention:
                return miu_pred[:, -self.pred_len:, :], Sigma_pred[:, -self.pred_len:, :, :], attns
            else:

                return miu_pred[:, -self.pred_len:, :], Sigma_pred[:, -self.pred_len:, :, :], dec_out  # [B, L, D]



    def compute_batch_xxT(batch_x):
        B, T, D = batch_x.shape
        device = batch_x.device
        batch_xxT = batch_x.unsqueeze(-1) @ batch_x.unsqueeze(-2) # [B, T, D, D]

        triu_indices = torch.triu_indices(D, D)
        #  shape [B, T, D*(D+1)//2]
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


    set_seed(514 + seed_idx)
    model_conditional_mean = ns_Transformer().float().to(device)
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


            if val_avg < best_val_loss and step >= num_training_steps//2:
                best_val_loss = val_avg
                best_step = step
                best_model_state = {k: v.clone() for k, v in model_conditional_mean.state_dict().items()}

        # print(f"{step}:Train={total_loss:.4f}|Val={val_avg:.4f}|Test={test_avg:.4f}|whiten_score_y={whiten_score_y:.4f}|whiten_score_y_cw={whiten_score_y_cw:.4f}|whiten_score_y_cent={whiten_score_y_centralized:.4f}")

    # ========== load best model ==========
    if best_model_state is not None:
        model_conditional_mean.load_state_dict(best_model_state)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")


    from interpretable_diffusion.gaussian_diffusion import *
    from interpretable_diffusion.model_utils_flowts import *
    from interpretable_diffusion.transformer_flowts import *
    import scipy
    import numpy as np

    from ts2vec.ts2vec import TS2Vec


    # metric
    class CRPS(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")


            # self.executor = ProcessPoolExecutor()
            

        def update(self, pred: torch.Tensor, true: torch.Tensor):
            print(pred.dtype,true.dtype)
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


    dataset_cw = ETTh1(root='ts_datasets')

    scaler_type = config_diffusion['scaler_type']
    ScalerClass = parse_type(scaler_type, globals())
    scaler = ScalerClass()

    windows = config['windows']
    horizon = config['horizon']
    pred_len = config['pred_len']
    batch_size = config['batch_size']
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
        batch_size=config_diffusion['batch_size'],
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


    future_cw_train = {}
    with torch.inference_mode():
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
            future_cw_train[i] = [batch_future.detach(), future_cw.detach(), cov_sqrt_inv.detach(), cov_sqrt.detach(), miu_pred.detach()]




    future_cw_val = {}
    with torch.inference_mode():
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
            future_cw_val[i] = [batch_future.detach(), future_cw.detach(), cov_sqrt_inv.detach(), cov_sqrt.detach(), miu_pred.detach()]




    future_cw_test = {}
    with torch.inference_mode():
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
            future_cw_test[i] = [batch_future.detach(), future_cw.detach(), cov_sqrt_inv.detach(), cov_sqrt.detach(), miu_pred.detach()]




    def fm_model_forward(model,
                        input_xt,
                        sigma,
                        cond,
                        x_t_mark,
                        history_mark,
                        mask=None,):

        model_output = model(input = input_xt, 
                            t = sigma.view(-1),
                            # history = cond,
                            # x_t_mark=x_t_mark, 
                            # history_mark=history_mark,
                            # cond_mask = mask
                            )

        return model_output

    def extract(input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)


    def upper_triangular_to_full_matrix(values: torch.Tensor, D: int) -> torch.Tensor:

        B, T, K = values.shape
        assert K == D*(D+1)//2, f"K={K} and D*(D+1)//2 not match (D={D})"

        idx = torch.triu_indices(D, D, offset=0, device=values.device)  # (2, K)

        v = values.reshape(B*T, K)
        mat = torch.zeros(B*T, D, D, device=values.device, dtype=values.dtype)
        mat[:, idx[0], idx[1]] = v

        mat = mat + mat.transpose(1, 2) - torch.diag_embed(torch.diagonal(mat, dim1=1, dim2=2))

        return mat.view(B, T, D, D)


    def fm_loss_fn(model, 
                    batch_x, 
                    batch_history, 
                    batch_x_mark,
                    batch_history_mark,
                    batch_cov_sqrt,
                    batch_miu,
                    num_features = num_features,
                    mask_cond_prob=0.5,
                    ot_beta = 0,
                    use_fft = config_diffusion['use_fft'],
                    device=device,):
        model.train()
        B, T, D = batch_x.shape
        batch_hisory_and_future = torch.cat([batch_history.to(device), batch_x.to(device)], dim = 1)
        sigma = torch.rand(B, 1, 1, device=batch_x.device) 
        noise = torch.randn_like(batch_hisory_and_future)

        cond_mean_future = batch_miu
        cond_cov_sqrt_future = batch_cov_sqrt


        noise[:,batch_history.shape[1]:,:] = torch.einsum('btij,btj->bti', cond_cov_sqrt_future, noise[:,batch_history.shape[1]:,:]) + cond_mean_future



        
        batch_x_t = sigma * batch_hisory_and_future + (1-sigma) * noise

        out = fm_model_forward(model=model,
                                input_xt= batch_x_t,
                                sigma= sigma,
                                cond= batch_history,
                                x_t_mark= batch_x_mark,
                                history_mark= batch_history_mark,
                                mask=None
                                )

        
        l2_loss = (batch_hisory_and_future-noise-out).square().mean()
        if use_fft:
            fft1 = torch.fft.fft(out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft((batch_hisory_and_future-noise).transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = (torch.real(fft1)- torch.real(fft2)).square().mean()\
                        + (torch.imag(fft1)- torch.imag(fft2)).square().mean()
            fourier_loss = use_fft * fourier_loss
        else: 
            fourier_loss = 0
        return l2_loss +  fourier_loss





    set_seed(514 + seed_idx)
    model_diffusion = Transformer(n_feat=num_features,
                                        n_channel = config_diffusion['pred_len'] + config_diffusion['windows'],
                                        n_layer_enc=config_diffusion['e_layers'],
                                        n_layer_dec=config_diffusion['d_layers'],
                                        n_embd=config_diffusion['d_model'],
                                        n_heads=config_diffusion['n_heads'],
                                        attn_pdrop=config_diffusion['dropout'],
                                        resid_pdrop=config_diffusion['dropout'],
                                        mlp_hidden_times=config_diffusion['mlp_hidden_times'],
                                        block_activate='GELU',
                                        max_len=config_diffusion['pred_len'] + config_diffusion['windows'],
                                        conv_params=None,).to(device)
    print(sum(p.numel() for p in model_diffusion.parameters()))
    optimizer = torch.optim.AdamW(
        model_diffusion.parameters(),
        lr=config_diffusion['lr'],
        weight_decay=config_diffusion['weight_decay'] 
    )


    val_loss = []
    num_training_steps = config_diffusion['num_training_steps']

    best_val_loss = float('inf')
    best_model_state = None
    best_step = -1

    for step in range(num_training_steps):
        model_diffusion.train()
        total_loss = 0
    ###################################################   train   #######################################
        for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader_cw.train_loader):


            optimizer.zero_grad()
            # future_cw_train[i] = 
            # [batch_future.detach(), future_cw.detach(), cov_sqrt_inv.detach(), cov_sqrt.detach(), miu_pred.detach()]



            batch_history = batch_x.to(device).float()
            batch_future = batch_y.to(device).float()
            batch_x_t_mark = batch_y_mark.to(device).float()
            batch_history_mark = batch_x_mark.to(device).float()

            loss = fm_loss_fn(model= model_diffusion, 
                            batch_x= batch_future, 
                            batch_history= batch_history, 
                            batch_x_mark=batch_x_t_mark,
                            batch_history_mark=batch_history_mark,
                            batch_cov_sqrt = future_cw_train[i][3].to(device).float(),
                            batch_miu = future_cw_train[i][4].to(device).float(),
                            mask_cond_prob=config_diffusion['mask_cond_prob'],
                            ot_beta = config_diffusion['ot_beta'],
                            use_fft = config_diffusion['use_fft'],
                            device=device,)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model_diffusion.parameters(), 1.)
            optimizer.step()


            total_loss += loss.item()
        total_loss = total_loss / len(dataloader_cw.train_loader)
        

    ###################################################   val   #######################################
        with torch.no_grad():
            model_diffusion.eval()
            val_total = 0
            for i, (batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_mark,
                batch_y_mark,
                ) in enumerate(dataloader_cw.val_loader):

                batch_history = batch_x.to(device).float()
                # batch_future = dataloader_cw.scaler.inverse_transform(batch_y.to(device)).float()
                batch_future = batch_y.to(device).float()
                batch_x_t_mark = batch_y_mark.to(device).float()
                batch_history_mark = batch_x_mark.to(device).float()

                loss = fm_loss_fn(model= model_diffusion, 
                            batch_x= batch_future, 
                            batch_history= batch_history, 
                            batch_x_mark=batch_x_t_mark,
                            batch_history_mark=batch_history_mark,
                            batch_cov_sqrt = future_cw_val[i][3].to(device).float(),
                            batch_miu = future_cw_val[i][4].to(device).float(),
                            mask_cond_prob=config_diffusion['mask_cond_prob'],
                            ot_beta = config_diffusion['ot_beta'],
                            use_fft = config_diffusion['use_fft'],
                            device=device,)
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
                batch_future = batch_y.to(device).float()
                batch_x_t_mark = batch_y_mark.to(device).float()
                batch_history_mark = batch_x_mark.to(device).float()
                
                test_loss = fm_loss_fn(model= model_diffusion, 
                            batch_x= batch_future, 
                            batch_history= batch_history, 
                            batch_x_mark=batch_x_t_mark,
                            batch_history_mark=batch_history_mark,
                            batch_cov_sqrt = future_cw_test[i][3].to(device).float(),
                            batch_miu = future_cw_test[i][4].to(device).float(),
                            mask_cond_prob=config_diffusion['mask_cond_prob'],
                            ot_beta = config_diffusion['ot_beta'],
                            use_fft = config_diffusion['use_fft'],
                            device=device,)

                test_total += test_loss.item()
                test_avg = test_total / len(dataloader_cw.test_loader)
            
            

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_step = step
                best_model_state = {k: v.clone() for k, v in model_diffusion.state_dict().items()}

        # print(f"Step {step}: Train Loss = {total_loss:.4f} | Val Loss = {val_avg:.4f} | Test Loss = {test_avg:.4f}")



    if best_model_state is not None:
        model_diffusion.load_state_dict(best_model_state)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")


    def fm_sampler(net, 
                latents, 
                history,
                x_t_mark,
                history_mark,
                cond_cov_sqrt_future,
                cond_mean_future,
                num_steps=50,
                adaptive_k = 0.1,
                second_correction=True):
        net.eval()
        traj = []
        his_mask = torch.ones_like(history)
        fut_mask = torch.zeros(history.shape[0], latents.shape[1] - history.shape[1], history.shape[2]).to(history.device)
        full_mask = torch.cat([his_mask.clone(), fut_mask.clone()], dim = 1).to(history.device)
        x_1_real = torch.cat([history.clone(), fut_mask.clone()], dim = 1).to(history.device)
        # Conditional mask (consistent with your current setup)
        cond_mask = None 

        #  t ∈ [0, 1]
        dt = 1.0 / float(num_steps)
        x_1_hat = latents
        traj.append(x_1_hat)

        for i in range(num_steps-1):
            x_0 = torch.randn_like(x_1_hat)
            x_0[:, history.shape[1]:, :] = torch.einsum('btij,btj->bti', cond_cov_sqrt_future, x_0[:, history.shape[1]:, :]) + cond_mean_future
            
            t_i = (i/num_steps) ** adaptive_k
            t_i = torch.full((x_1_hat.shape[0],1,1), t_i, device=x_1_hat.device, dtype=torch.float32)

            x_1_hat = (1-full_mask) * x_1_hat + full_mask * x_1_real
            # print(t_i.shape , x_1_hat.shape, x_0.shape)

            x_ti_hat = t_i * x_1_hat + (1 - t_i) * x_0

            # f(t, x_t): -
            v_t = fm_model_forward(model=net,
                                input_xt=x_ti_hat,
                                sigma=t_i.view(-1),
                                cond=history,
                                x_t_mark=x_t_mark,
                                history_mark=history_mark,
                                mask=cond_mask)
            
            x_1_hat = x_ti_hat + (1-t_i) * v_t



            traj.append(x_1_hat)

        return traj


    for i, (batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_mark,
        batch_y_mark,
        ) in enumerate(dataloader_cw.test_loader):
        batch_history = batch_x.to(device).float()
        batch_future = batch_y.to(device).float()
        batch_x_t_mark = batch_y_mark.to(device).float()
        batch_history_mark = batch_x_mark.to(device).float()


    for i, (batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_mark,
        batch_y_mark,
        ) in enumerate(dataloader.test_loader):
        batch_history_original = batch_x.to(device).float()
        batch_future_original = batch_y.to(device).float()
        batch_future_mark_original = batch_y_mark.to(device).float()
        batch_history_mark_original = batch_x_mark.to(device).float()


    generated_batch = []
    num_steps = 5
    use_cond = True
    num = 100
    second_correction=True

    cond_mean_future = future_cw_test[0][4].to(device).float()
    cond_cov_sqrt_future = future_cw_test[0][3].to(device).float()




    set_seed(1920+ seed_idx)
    from tqdm import tqdm
    sample_device = device
    with torch.inference_mode():
        for iii in tqdm(range(num)):
            latents = torch.randn_like(batch_future[:,:,:num_features]).to(sample_device)
            latents = torch.einsum('btij,btj->bti', cond_cov_sqrt_future, latents) + cond_mean_future
            latents = torch.cat([torch.randn_like(batch_history).to(sample_device), latents],dim = 1)

            
            if use_cond:
                samples = fm_sampler(net=model_diffusion.to(sample_device), 
                                    latents = latents.to(sample_device), 
                                    history = batch_history[:,:,:num_features].to(sample_device),
                                    x_t_mark= batch_x_t_mark.to(sample_device),
                                    history_mark= batch_history_mark.to(sample_device),
                                    cond_cov_sqrt_future = cond_cov_sqrt_future,
                                    cond_mean_future = cond_mean_future,
                                    num_steps=num_steps,
                                    second_correction=True)
            else:
                samples = fm_sampler(net=model_diffusion.to(sample_device), 
                                    latents = latents.to(sample_device), 
                                    history = None,
                                    x_t_mark= batch_x_t_mark.to(sample_device),
                                    history_mark= batch_history_mark.to(sample_device),
                                    cond_cov_sqrt_future = cond_cov_sqrt_future,
                                    cond_mean_future = cond_mean_future,
                                    num_steps=num_steps,
                                    second_correction=True)
            generated_batch.append(samples[-1][:,batch_history.shape[1]:,:].unsqueeze(-1))


    generated_batch = [
        x.to(device, non_blocking=True) if torch.is_tensor(x) else x
        for x in generated_batch
    ]





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
    metrics.update(torch.cat(generated_batch,dim=-1).float().detach().cpu(), batch_future_original.detach().cpu())


    metrics_dict = {name: float(metric.compute()) for name, metric in metrics.items()}


    pred_xy = [torch.cat([batch_history_original,x.squeeze(-1)],dim=1) for x in generated_batch]
    pred_xy = torch.cat(pred_xy,dim=0)


    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device:
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    set_seed(810 + seed_idx)
    context_fid = Context_FID(ori_data = torch.cat([batch_history_original,batch_future_original],dim=1).detach().cpu().numpy(), 
                generated_data = pred_xy.detach().cpu().numpy())


    cacf_list = []
    for iii in range(len(generated_batch)):
        cacf_list.append((cacf_torch(x = torch.cat([batch_history_original,batch_future_original],dim=1)) \
                        -cacf_torch(x =  torch.cat([batch_history_original,generated_batch[iii].squeeze(-1)],dim=1))).abs().mean())
    cacf_mean, cacf_std = torch.tensor(cacf_list).mean(),torch.tensor(cacf_list).std()


    lambda_min = config['eps_eign_min']
    weigen = config['eign_penalty']
    window_length = config['window_size']
    log_file = f'cw_flowts_lambdamin_{lambda_min}_weigen{weigen}.txt'
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


