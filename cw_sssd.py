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
    from einops import rearrange, repeat


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

    data_path = "ts_datasets/ETTh1.csv"
    assert os.path.exists(data_path), f"Data file not found: {data_path}"

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


    # we can see, the std is very big. so, we have to do: 
    # batch_xxT / train_set_xxT_std, and batch_yyT / train_set_xxT_std
    # batch_xxT_trig / train_set_xxT_up_trig_std, and batch_yyT_trig / train_set_xxT_up_trig_std. 
    # and to keep the diagnol element be positive, we should not - mean.
    # train_set_xxT_up_trig_std


    config = {
        # 
        "teacher_force": False,
        "mixup": False,

        # loss 
        "matrix_norm_weight": [(7*192)**0.5 * 0.05, 1, 0.],
        "fft_weight": [1, 0.],
        "eign_penalty": 50,
        "eps_eign_min": 0.05,
        "penalty_method": "hard",
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

        # 
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

    lambda_min = config['eps_eign_min']
    weigen = config['eign_penalty']
    window_length = config['window_size']
    log_file = f"cw_sssd_lambda_{lambda_min}_weigen{weigen}.txt"

    config_diffusion = {
        
        # diffusion 
        'diff_total_steps': 1000,
        'noise_schedule': 'linear',
        'noise_start': 1e-4,
        'noise_end': 2e-2,
        
        # S4 
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 3,
        "d_layers": 3,
        "mlp_hidden_times": 4,
        "dropout": 0.1,
        'd_state': 64,
        'l_max': 200,
        'mask_prob': 0.75,

        # 
        "windows": 168,
        "horizon": 1,
        "pred_len": 192,
        "label_len": 168 // 2,

        # DataLoader 
        "batch_size": 128,
        "num_worker": 0,
        'dataset_type': "ETTh1",
        'data_path': "ts_datasets",
        'scaler_type': "StandardScaler",
        
        # 
        'num_training_steps': 20,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'use_fft': 192**0.5 / 5,

    }


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





    # === 1. 
    dataset_type = config['dataset_type']
    data_path = config['data_path']

    DatasetClass = parse_type(dataset_type, globals())
    dataset = DatasetClass(root=data_path)

    # === 2.  scaler
    scaler_type = config['scaler_type']
    ScalerClass = parse_type(scaler_type, globals())
    scaler = ScalerClass()

    # === 3.  dataloader
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


    # dataloader.train_loader.__len__(),dataloader.val_loader.__len__(),dataloader.test_loader.__len__()


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
            self.pred_len = pred_len # 
            self.seq_len = seq_len # 
            self.label_len = label_len # 
            self.output_attention = output_attention # 
            self.num_feature = num_features
            self.num_feature_triangle = int(num_features*(num_features+1)/2)

            # Embedding （B,T,N)
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                            dropout) # 
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                            dropout) # 
            # Encoder： ，
            # DSAttention，EncoderLayer
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

            # Decoder：，、
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
                # 
                projection=nn.Linear(d_model, c_out, bias=True)
            )

            # 
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
            U[:, :, range(D), range(D)] = F.softplus(U[:, :, range(D), range(D)])  # 
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

            # （label_len)0(pred_len)
            x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                                dim=1).to(x_enc.device).clone()

            # Projector（）
            tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
            delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S


            # 
            enc_out = self.enc_embedding(x_enc, x_mark_enc)

            
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

            # 
            dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

            # De-normalization
            # dec_out = dec_out * std_enc + mean_enc
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

        # 
        mat = torch.zeros(B, dim, dim, device=values.device)

        # 
        mat[:, idx[0], idx[1]] = values.squeeze(1)

        # ，
        mat = mat + mat.transpose(1,2) - torch.diag_embed(torch.diagonal(mat, dim1=1, dim2=2))
        return mat


    def upper_triangular_to_full_matrix(values, dim):

        B = values.shape[0]
        num_pairs = values.shape[2]
        idx = torch.triu_indices(row=dim, col=dim, offset=0, device=values.device)

        # 
        mat = torch.zeros(B, dim, dim, device=values.device)

        # （）
        mat[:, idx[0], idx[1]] = values.squeeze(1)

        # （）
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
        """
         softplus(β*(eps-λ))/β，。
        eigvals: [B, T, D]
        return : 
        """
        eps_t = torch.as_tensor(eps, dtype=eigvals.dtype, device=eigvals.device)
        gaps  = eps_t - eigvals                 # [B,T,D]
        softg = F.softplus(beta * gaps) / beta  # “”
        return softg.mean()                     # （）


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
        # 
        loss_miu = F.mse_loss(miu_pred, theta_true)
        fft_loss_miu = (torch.fft.rfft(miu_pred, dim=1) - torch.fft.rfft(theta_true, dim=1)).abs().mean()

        ######################### dont forget to normalize the sigma here! #######################
        diff = (sigma_pred - theta_outer_true) / batch_xxT_std
        loss_fro = diff.pow(2).mean()
        svals = torch.linalg.svdvals(diff)  # [B, T, D]
        loss_svd = svals.mean()
        

        # ： Σ - μμᵀ 0
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

                # 
                eigvals, eigvecs = torch.linalg.eigh(cov)

                if verbose and (eigvals < eps).any():
                    print(f"eigvals before clamp: {eigvals}")

                # （ ≥ eps）
                eigvals_clamped = torch.clamp(eigvals, min=eps)

                # inverse sqrt
                inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals_clamped)) @ eigvecs.T
                inv_sqrt_all[b, t] = inv_sqrt
                whitened[b, t] = inv_sqrt @ residual[b, t]

                # sqrt
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

    # 
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

            # ===  mixup ===
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
            # ===  mixup ===

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

            

            # 
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_step = step
                best_model_state = {k: v.clone() for k, v in model_conditional_mean.state_dict().items()}

        # print(f"{step}:Train={total_loss:.4f}|Val={val_avg:.4f}|Test={test_avg:.4f}|whiten_score_y={whiten_score_y:.4f}|whiten_score_y_cw={whiten_score_y_cw:.4f}|whiten_score_y_cent={whiten_score_y_centralized:.4f}")

    # ==========  ==========
    if best_model_state is not None:
        model_conditional_mean.load_state_dict(best_model_state)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")


    from interpretable_diffusion.gaussian_diffusion import *
    from interpretable_diffusion.model_utils import *
    from interpretable_diffusion.transformer import *
    import scipy
    import numpy as np

    from ts2vec.ts2vec import TS2Vec





    # from interpretable_diffusion.gaussian_diffusion import *
    # from interpretable_diffusion.model_utils import *
    # from interpretable_diffusion.transformer import *
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

    def Context_FID(ori_data, test_data, generated_data):
        model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=512, lr=0.001, output_dims=320,
                    max_train_length=3000)
        model.fit(ori_data, verbose=False, )
        ori_represenation = model.encode(test_data, encoding_window='full_series')
        gen_represenation = model.encode(generated_data, encoding_window='full_series')
        idx = np.random.permutation(ori_data.shape[0])
        ori_represenation = ori_represenation[idx]
        gen_represenation = gen_represenation[idx]
        results = calculate_fid(ori_represenation, gen_represenation)
        return results


    dataset_cw = ETTh1(root='ts_datasets')
    # === 2.  scaler
    scaler_type = config_diffusion['scaler_type']
    ScalerClass = parse_type(scaler_type, globals())
    scaler = ScalerClass()

    # === 3.  dataloader
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

        #  [B,T,D,D]
        sqrt_diag = torch.diag_embed(sqrt_vals)        # [B,T,D,D]
        inv_sqrt_diag = torch.diag_embed(inv_sqrt_vals)

        # ---  (vecs @ diag @ vecs^T) ---
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

            # ：detach  xxT / xxT_trig 
            batch_his_xxT, batch_his_xxT_trig = batch_history_sliding_cov.detach(), batch_history_sliding_cov_trig.detach()
            batch_fur_xxT, batch_fur_xxT_trig = batch_future_sliding_cov.detach(), batch_future_sliding_cov_trig.detach()

            # --- decoder （） ---
            #  zero（pred_len, N + n_up_trig）
            # dec_inp_pred = torch.zeros(
            #     (batch_history.shape[0], pred_len, num_features ), dtype=batch_future.dtype, device=device
            # )
            dec_inp_pred = torch.zeros(
                [batch_history.size(0), pred_len, num_features + int(num_features*(num_features+1)/2)]
            ).to(device)
            # label ：x  label_len  + 
            # dec_inp_label = batch_history[:, -label_len:, :]
            dec_inp_label = torch.cat([batch_history[:, -label_len :, :].to(device),batch_his_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
            # print(dec_inp_label.shape, dec_inp_pred.shape)
            dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)  # [1, label_len+pred_len, N+n_up_trig]

            batch_future_mark_input = torch.concat([batch_history_mark[:, -label_len:, :], batch_future_mark], dim=1)

            # ---  ---
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

            # ：detach  xxT / xxT_trig 
            batch_his_xxT, batch_his_xxT_trig = batch_history_sliding_cov.detach(), batch_history_sliding_cov_trig.detach()
            batch_fur_xxT, batch_fur_xxT_trig = batch_future_sliding_cov.detach(), batch_future_sliding_cov_trig.detach()

            # --- decoder （） ---
            #  zero（pred_len, N + n_up_trig）
            dec_inp_pred = torch.zeros(
                [batch_history.size(0), pred_len, num_features + int(num_features*(num_features+1)/2)]
            ).to(device)
            # label ：x  label_len  + 
            # dec_inp_label = batch_history[:, -label_len:, :]
            dec_inp_label = torch.cat([batch_history[:, -label_len :, :].to(device),batch_his_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
            # print(dec_inp_label.shape, dec_inp_pred.shape)
            dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)  # [1, label_len+pred_len, N+n_up_trig]

            batch_future_mark_input = torch.concat([batch_history_mark[:, -label_len:, :], batch_future_mark], dim=1)

            # ---  ---
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

            # ：detach  xxT / xxT_trig 
            batch_his_xxT, batch_his_xxT_trig = batch_history_sliding_cov.detach(), batch_history_sliding_cov_trig.detach()
            batch_fur_xxT, batch_fur_xxT_trig = batch_future_sliding_cov.detach(), batch_future_sliding_cov_trig.detach()

            # --- decoder （） ---
            #  zero（pred_len, N + n_up_trig）
            dec_inp_pred = torch.zeros(
                [batch_history.size(0), pred_len, num_features + int(num_features*(num_features+1)/2)]
            ).to(device)
            # label ：x  label_len  + 
            # dec_inp_label = batch_history[:, -label_len:, :]
            dec_inp_label = torch.cat([batch_history[:, -label_len :, :].to(device),batch_his_xxT_trig[:, -label_len:, :].to(device)],dim=-1)
            # print(dec_inp_label.shape, dec_inp_pred.shape)
            dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)  # [1, label_len+pred_len, N+n_up_trig]

            batch_future_mark_input = torch.concat([batch_history_mark[:, -label_len:, :], batch_future_mark], dim=1)

            # ---  ---
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




    import opt_einsum as oe

    contract = oe.contract
    contract_expression = oe.contract_expression
    ''' Standalone CSDI + S4 imputer for random missing, non-random missing and black-out missing.
    The notebook contains CSDI and S4 functions and utilities. However the imputer is located in the last Class of
    the notebook, please see more documentation of use there. Additional at this file can be added for CUDA multiplication 
    the cauchy kernel.'''

    # def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    #     """Initializes multi-GPU-friendly python logger."""

    #     logger = logging.getLogger(name)
    #     logger.setLevel(level)

    #     # this ensures all logging levels get marked with the rank zero decorator
    #     # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    #     for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
    #         setattr(logger, level, rank_zero_only(getattr(logger, level)))

    #     return logger
    # log = get_logger(__name__)

    """ Cauchy kernel """

    try: # Try CUDA extension
        from extensions.cauchy.cauchy import cauchy_mult
        has_cauchy_extension = True
    except:
        print(
            "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
        )
        has_cauchy_extension = False


    has_pykeops = False
    if not has_cauchy_extension:
        print(
            "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for efficiency."
        )
        def cauchy_slow(v, z, w):
            """
            v, w: (..., N)
            z: (..., L)
            returns: (..., L)
            """
            cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1) + 1e-6) # (... N L)
            return torch.sum(cauchy_matrix, dim=-2)

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    _c2r = torch.view_as_real
    _r2c = torch.view_as_complex
    _conj = lambda x: torch.cat([x, x.conj()], dim=-1)
    if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
        _resolve_conj = lambda x: x.conj().resolve_conj()
    else:
        _resolve_conj = lambda x: x.conj()



    """ simple nn.Module components """

    def Activation(activation=None, dim=-1):
        if activation in [ None, 'id', 'identity', 'linear' ]:
            return nn.Identity()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation in ['swish', 'silu']:
            return nn.SiLU()
        elif activation == 'glu':
            return nn.GLU(dim=dim)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

    def get_initializer(name, activation=None):
        if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
            nonlinearity = 'linear'
        elif activation in ['relu', 'tanh', 'sigmoid']:
            nonlinearity = activation
        elif activation in ['gelu', 'swish', 'silu']:
            nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
        else:
            raise NotImplementedError(f"get_initializer: activation {activation} not supported")

        if name == 'uniform':
            initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
        elif name == 'normal':
            initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
        elif name == 'xavier':
            initializer = torch.nn.init.xavier_normal_
        elif name == 'zero':
            initializer = partial(torch.nn.init.constant_, val=0)
        elif name == 'one':
            initializer = partial(torch.nn.init.constant_, val=1)
        else:
            raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

        return initializer

    class TransposedLinear(nn.Module):
        """ Linear module on the second-to-last dimension """

        def __init__(self, d_input, d_output, bias=True):
            super().__init__()

            self.weight = nn.Parameter(torch.empty(d_output, d_input))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
            # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

            if bias:
                self.bias = nn.Parameter(torch.empty(d_output, 1))
                bound = 1 / math.sqrt(d_input+ 1e-6)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.bias = 0.0

        def forward(self, x):
            return contract('... u l, v u -> ... v l', x, self.weight) + self.bias

    def LinearActivation(
            d_input, d_output, bias=True,
            zero_bias_init=False,
            transposed=False,
            initializer=None,
            activation=None,
            activate=False, # Apply activation as part of this module
            weight_norm=False,
            **kwargs,
        ):
        """ Returns a linear nn.Module with control over axes order, initialization, and activation """

        # Construct core module
        linear_cls = TransposedLinear if transposed else nn.Linear
        if activation == 'glu': d_output *= 2
        linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

        # Initialize weight
        if initializer is not None:
            get_initializer(initializer, activation)(linear.weight)

        # Initialize bias
        if bias and zero_bias_init:
            nn.init.zeros_(linear.bias)

        # Weight norm
        if weight_norm:
            linear = nn.utils.weight_norm(linear)

        if activate and activation is not None:
            activation = Activation(activation, dim=-2 if transposed else -1)
            linear = nn.Sequential(linear, activation)
        return linear



    """ Misc functional utilities """

    def krylov(L, A, b, c=None, return_power=False):
        """
        Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

        If return_power=True, return A^{L-1} as well
        """
        # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

        x = b.unsqueeze(-1) # (..., N, 1)
        A_ = A

        AL = None
        if return_power:
            AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
            _L = L-1

        done = L == 1
        # loop invariant: _L represents how many indices left to compute
        while not done:
            if return_power:
                if _L % 2 == 1: AL = A_ @ AL
                _L //= 2

            # Save memory on last iteration
            l = x.shape[-1]
            if L - l <= l:
                done = True
                _x = x[..., :L-l]
            else: _x = x

            _x = A_ @ _x
            x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
            if not done: A_ = A_ @ A_

        assert x.shape[-1] == L

        if c is not None:
            x = torch.einsum('...nl, ...n -> ...l', x, c)
        x = x.contiguous() # WOW!!
        if return_power:
            return x, AL
        else:
            return x

    def power(L, A, v=None):
        """ Compute A^L and the scan sum_i A^i v_i

        A: (..., N, N)
        v: (..., N, L)
        """

        I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

        powers = [A]
        l = 1
        while True:
            if L % 2 == 1: I = powers[-1] @ I
            L //= 2
            if L == 0: break
            l *= 2
            powers.append(powers[-1] @ powers[-1])

        if v is None: return I

        # Invariants:
        # powers[-1] := A^l
        # l := largest po2 at most L

        # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
        # We do this reverse divide-and-conquer for efficiency reasons:
        # 1) it involves fewer padding steps for non-po2 L
        # 2) it involves more contiguous arrays

        # Take care of edge case for non-po2 arrays
        # Note that this initial step is a no-op for the case of power of 2 (l == L)
        k = v.size(-1) - l
        v_ = powers.pop() @ v[..., l:]
        v = v[..., :l]
        v[..., :k] = v[..., :k] + v_

        # Handle reduction for power of 2
        while v.size(-1) > 1:
            v = rearrange(v, '... (z l) -> ... z l', z=2)
            v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
        return I, v.squeeze(-1)


    """ HiPPO utilities """

    def embed_c2r(A):
        A = rearrange(A, '... m n -> ... m () n ()')
        A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
            np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
        return rearrange(A, 'm x n y -> (m x) (n y)')

    def transition(measure, N, **measure_args):
        """ A, B transition matrices for different measures

        measure: the type of measure
        legt - Legendre (translated)
        legs - Legendre (scaled)
        glagt - generalized Laguerre (translated)
        lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
        """
        # Laguerre (translated)
        if measure == 'lagt':
            b = measure_args.get('beta', 1.0)
            A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
            B = b * np.ones((N, 1))
        # Generalized Laguerre
        # alpha 0, beta small is most stable (limits to the 'lagt' measure)
        # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
        elif measure == 'glagt':
            alpha = measure_args.get('alpha', 0.0)
            beta = measure_args.get('beta', 0.01)
            A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
            B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

            L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
            A = (1./L[:, None]) * A * L[None, :]
            B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
        # Legendre (translated)
        elif measure == 'legt':
            Q = np.arange(N, dtype=np.float64)
            R = (2*Q + 1) ** .5
            j, i = np.meshgrid(Q, Q)
            A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
            B = R[:, None]
            A = -A
        # Legendre (scaled)
        elif measure == 'legs':
            q = np.arange(N, dtype=np.float64)
            col, row = np.meshgrid(q, q)
            r = 2 * q + 1
            M = -(np.where(row >= col, r, 0) - np.diag(q))
            T = np.sqrt(np.diag(2 * q + 1))
            A = T @ M @ np.linalg.inv(T)
            B = np.diag(T)[:, None]
            B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        elif measure == 'fourier':
            freqs = np.arange(N//2)
            d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
            A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
            A = A - embed_c2r(np.ones((N//2, N//2)))
            B = embed_c2r(np.ones((N//2, 1)))[..., :1]
        elif measure == 'random':
            A = np.random.randn(N, N) / N
            B = np.random.randn(N, 1)
        elif measure == 'diagonal':
            A = -np.diag(np.exp(np.random.randn(N)))
            B = np.random.randn(N, 1)
        else:
            raise NotImplementedError

        return A, B

    def rank_correction(measure, N, rank=1, dtype=torch.float):
        """ Return low-rank matrix L such that A + L is normal """

        if measure == 'legs':
            assert rank >= 1
            P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
        elif measure == 'legt':
            assert rank >= 2
            P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
            P0 = P.clone()
            P0[0::2] = 0.
            P1 = P.clone()
            P1[1::2] = 0.
            P = torch.stack([P0, P1], dim=0) # (2 N)
        elif measure == 'lagt':
            assert rank >= 1
            P = .5**.5 * torch.ones(1, N, dtype=dtype)
        elif measure == 'fourier':
            P = torch.ones(N, dtype=dtype) # (N)
            P0 = P.clone()
            P0[0::2] = 0.
            P1 = P.clone()
            P1[1::2] = 0.
            P = torch.stack([P0, P1], dim=0) # (2 N)
        else: raise NotImplementedError

        d = P.size(0)
        if rank > d:
            P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
        return P

    def nplr(measure, N, rank=1, dtype=torch.float):
        """ Return w, p, q, V, B such that
        (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
        i.e. A = V[w - p q^*]V^*, B = V B
        """
        assert dtype == torch.float or torch.cfloat
        if measure == 'random':
            dtype = torch.cfloat if dtype == torch.float else torch.cdouble
            # w = torch.randn(N//2, dtype=dtype)
            w = -torch.exp(torch.randn(N//2)) + 1j*torch.randn(N//2)
            P = torch.randn(rank, N//2, dtype=dtype)
            B = torch.randn(N//2, dtype=dtype)
            V = torch.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
            return w, P, B, V

        A, B = transition(measure, N)
        A = torch.as_tensor(A, dtype=dtype) # (N, N)
        B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

        P = rank_correction(measure, N, rank=rank, dtype=dtype)
        AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)
        w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
        # V w V^{-1} = A

        # Only keep one of the conjugate pairs
        w = w[..., 0::2].contiguous()
        V = V[..., 0::2].contiguous()

        V_inv = V.conj().transpose(-1, -2)

        B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
        P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P


        return w, P, B, V


    def bilinear(dt, A, B=None):
        """
        dt: (...) timescales
        A: (... N N)
        B: (... N)
        """
        N = A.shape[-1]
        I = torch.eye(N).to(A)
        A_backwards = I - dt[:, None, None] / 2 * A
        A_forwards = I + dt[:, None, None] / 2 * A

        if B is None:
            dB = None
        else:
            dB = dt[..., None] * torch.linalg.solve(
                A_backwards, B.unsqueeze(-1)
            ).squeeze(-1) # (... N)

        dA = torch.linalg.solve(A_backwards, A_forwards)  # (... N N)
        return dA, dB




    class SSKernelNPLR(nn.Module):
        """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

        The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
        The parameters of this function are as follows.

        A: (... N N) the state matrix
        B: (... N) input matrix
        C: (... N) output matrix
        dt: (...) timescales / discretization step size
        p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

        The forward pass of this Module returns:
        (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

        """

        @torch.no_grad()
        def _setup_C(self, double_length=False):
            """ Construct C~ from C

            double_length: current C is for length L, convert it to length 2L
            """
            C = _r2c(self.C)
            self._setup_state()
            dA_L = power(self.L, self.dA)
            # Multiply C by I - dA_L
            C_ = _conj(C)
            prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
            if double_length: prod = -prod # Multiply by I + dA_L instead
            C_ = C_ - prod
            C_ = C_[..., :self.N] # Take conjugate pairs again
            self.C.copy_(_c2r(C_))

            if double_length:
                self.L *= 2
                self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

        def _omega(self, L, dtype, device, cache=True):
            """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
            This should be called everytime the internal length self.L changes """
            omega = torch.tensor(
                np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
            )  # \omega_{2L}
            omega = omega ** torch.arange(0, L // 2 + 1, device=device)
            z = 2 * (1 - omega) / (1 + omega)
            if cache:
                self.register_buffer("omega", _c2r(omega))
                self.register_buffer("z", _c2r(z))
            return omega, z

        def __init__(
            self,
            L, w, P, B, C, log_dt,
            hurwitz=False,
            trainable=None,
            lr=None,
            tie_state=False,
            length_correction=True,
            verbose=False,
        ):
            """
            L: Maximum length; this module computes an SSM kernel of length L
            w: (N)
            p: (r, N) low-rank correction to A
            q: (r, N)
            A represented by diag(w) - pq^*

            B: (N)
            dt: (H) timescale per feature
            C: (H, C, N) system is 1-D to c-D (channels)

            hurwitz: tie pq and ensure w has negative real part
            trainable: toggle which of the parameters is trainable
            lr: add hook to set lr of hippo parameters specially (everything besides C)
            tie_state: tie all state parameters across the H hidden features
            length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

            Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
            """

            super().__init__()
            self.hurwitz = hurwitz
            self.tie_state = tie_state
            self.verbose = verbose

            # Rank of low-rank correction
            self.rank = P.shape[-2]
            assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
            self.H = log_dt.size(-1)
            self.N = w.size(-1)

            # Broadcast everything to correct shapes
            C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)
            H = 1 if self.tie_state else self.H
            B = repeat(B, 'n -> 1 h n', h=H)
            P = repeat(P, 'r n -> r h n', h=H)
            w = repeat(w, 'n -> h n', h=H)

            # Cache Fourier nodes every time we set up a desired length
            self.L = L
            if self.L is not None:
                self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

            # Register parameters
            # C is a regular parameter, not state
            # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))
            self.C = nn.Parameter(_c2r(_resolve_conj(C)))
            train = False
            if trainable is None: trainable = {}
            if trainable == False: trainable = {}
            if trainable == True: trainable, train = {}, True
            self.register("log_dt", log_dt, trainable.get('dt', train), lr, 0.0)
            self.register("B", _c2r(B), trainable.get('B', train), lr, 0.0)
            self.register("P", _c2r(P), trainable.get('P', train), lr, 0.0)
            if self.hurwitz:
                log_w_real = torch.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
                w_imag = w.imag
                self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
                self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)
                self.Q = None
            else:
                self.register("w", _c2r(w), trainable.get('A', train), lr, 0.0)
                # self.register("Q", _c2r(P.clone().conj().resolve_conj()), trainable.get('P', train), lr, 0.0)
                Q = _resolve_conj(P.clone())
                self.register("Q", _c2r(Q), trainable.get('P', train), lr, 0.0)

            if length_correction:
                self._setup_C()

        def _w(self):
            # Get the internal w (diagonal) parameter
            if self.hurwitz:
                w_real = -torch.exp(self.log_w_real)
                w_imag = self.w_imag
                w = w_real + 1j * w_imag
            else:
                w = _r2c(self.w)  # (..., N)
            return w

        def forward(self, state=None, rate=1.0, L=None):
            """
            state: (..., s, N) extra tensor that augments B
            rate: sampling rate factor
            L: target output length ().  None， rate  self.L 

            returns: (..., c+s, L)
            """
            # Handle sampling rate logic
            assert not (rate is None and L is None)
            if rate is None:
                rate = self.L / L
            if L is None:
                L = int(self.L / rate)

            # ====== ： L >= 1， k_f.shape[-1] == 0 ======
            L = max(1, int(L))  # ✅  L  1， irfft  0
            # ============================================================

            # Increase the internal length if needed
            while rate * L > self.L:
                self.double_length()

            dt = torch.exp(self.log_dt) * rate
            B = _r2c(self.B)
            C = _r2c(self.C)
            P = _r2c(self.P)
            Q = P.conj() if self.Q is None else _r2c(self.Q)
            w = self._w()

            if rate == 1.0:
                omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
            else:
                omega, z = self._omega(int(self.L / rate), dtype=w.dtype, device=w.device, cache=False)

            if self.tie_state:
                B = repeat(B, '... 1 n -> ... h n', h=self.H)
                P = repeat(P, '... 1 n -> ... h n', h=self.H)
                Q = repeat(Q, '... 1 n -> ... h n', h=self.H)

            # Augment B
            if state is not None:
                s = _conj(state) if state.size(-1) == self.N else state
                sA = (
                    s * _conj(w)
                    - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
                )
                s = s / dt.unsqueeze(-1) + sA / 2
                s = s[..., :self.N]
                B = torch.cat([s, B], dim=-3)

            w = w * dt.unsqueeze(-1)
            B = torch.cat([B, P], dim=-3)
            C = torch.cat([C, Q], dim=-3)

            v = B.unsqueeze(-3) * C.unsqueeze(-4)

            if has_cauchy_extension and z.dtype == torch.cfloat:
                r = cauchy_mult(v, z, w, symmetric=True)
            elif has_pykeops:
                r = cauchy_conj(v, z, w)
            else:
                r = cauchy_slow(v, z, w)

            r = r * dt[None, None, :, None]

            # Low-rank Woodbury correction
            if self.rank == 1:
                k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
            elif self.rank == 2:
                r00 = r[: -self.rank, : -self.rank, :, :]
                r01 = r[: -self.rank, -self.rank :, :, :]
                r10 = r[-self.rank :, : -self.rank, :, :]
                r11 = r[-self.rank :, -self.rank :, :, :]
                det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
                s = (
                    r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                    + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                    - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                    - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
                )
                s = s / det
                k_f = r00 - s
            else:
                r00 = r[:-self.rank, :-self.rank, :, :]
                r01 = r[:-self.rank, -self.rank:, :, :]
                r10 = r[-self.rank:, :-self.rank, :, :]
                r11 = r[-self.rank:, -self.rank:, :, :]
                r11 = rearrange(r11, "a b h n -> h n a b")
                r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
                r11 = rearrange(r11, "h n a b -> a b h n")
                k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

            # Final correction for the bilinear transform
            k_f = k_f * 2 / (1 + omega)

            # ====== ： k_f （） 0！======
            if k_f.shape[-1] <= 0:
                # ⚠️  k_f  0（！）， dummy 
                # ， freq_dim=1， 1 ，
                print("[SSKernelNPLR] ⚠️ Warning: k_f.shape[-1] == 0, using fallback dummy k_f.")
                dummy_L = 1  # 
                k_f = torch.zeros(*k_f.shape[:-1], dummy_L, dtype=k_f.dtype, device=k_f.device)  # [S+1, C, H, 1]
            # ================================================================

            # Move from frequency to coefficients
            k = torch.fft.irfft(k_f)  # (S+1, C, H, L) ✅ ！

            # Truncate to target length
            k = k[..., :L]

            if state is not None:
                k_state = k[:-1, :, :, :]
            else:
                k_state = None
            k_B = k[-1, :, :, :]
            return k_B, k_state

        @torch.no_grad()
        def double_length(self):
            if self.verbose: print(f"S4: Doubling length from L = {self.L} to {2*self.L}")
            self._setup_C(double_length=True)

        def _setup_linear(self):
            """ Create parameters that allow fast linear stepping of state """
            w = self._w()
            B = _r2c(self.B) # (H N)
            P = _r2c(self.P)
            Q = P.conj() if self.Q is None else _r2c(self.Q)

            # Prepare Linear stepping
            dt = torch.exp(self.log_dt)
            D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
            R = (torch.eye(self.rank, dtype=w.dtype, device=w.device) + 2*contract('r h n, h n, s h n -> h r s', Q, D, P).real) # (H r r)
            Q_D = rearrange(Q*D, 'r h n -> h r n')
            R = torch.linalg.solve(R.to(Q_D), Q_D) # (H r N)
            R = rearrange(R, 'h r n -> r h n')

            self.step_params = {
                "D": D, # (H N)
                "R": R, # (r H N)
                "P": P, # (r H N)
                "Q": Q, # (r H N)
                "B": B, # (1 H N)
                "E": 2.0 / dt.unsqueeze(-1) + w, # (H N)
            }

        def _step_state_linear(self, u=None, state=None):
            """
            Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

            Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

            u: (H) input
            state: (H, N/2) state with conjugate pairs
            Optionally, the state can have last dimension N
            Returns: same shape as state
            """
            C = _r2c(self.C) # View used for dtype/device

            if u is None: # Special case used to find dA
                u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
            if state is None: # Special case used to find dB
                state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

            step_params = self.step_params.copy()
            if state.size(-1) == self.N: # Only store half of the conjugate pairs; should be true by default
                # There should be a slightly faster way using conjugate symmetry
                contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
            else:
                assert state.size(-1) == 2*self.N
                step_params = {k: _conj(v) for k, v in step_params.items()}
                # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
                contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
            D = step_params["D"]  # (H N)
            E = step_params["E"]  # (H N)
            R = step_params["R"]  # (r H N)
            P = step_params["P"]  # (r H N)
            Q = step_params["Q"]  # (r H N)
            B = step_params["B"]  # (1 H N)

            new_state = E * state - contract_fn(P, Q, state) # (B H N)
            new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
            new_state = D * (new_state - contract_fn(P, R, new_state))

            return new_state

        def _setup_state(self):
            """ Construct dA and dB for discretized state equation """

            # Construct dA and dB by using the stepping
            self._setup_linear()
            C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

            state = torch.eye(2*self.N, dtype=C.dtype, device=C.device).unsqueeze(-2) # (N 1 N)
            dA = self._step_state_linear(state=state)
            dA = rearrange(dA, "n h m -> h m n")
            self.dA = dA # (H N N)

            u = C.new_ones(self.H)
            dB = self._step_state_linear(u=u)
            dB = _conj(dB)
            self.dB = rearrange(dB, '1 h n -> h n') # (H N)

        def _step_state(self, u, state):
            """ Must be called after self.default_state() is used to construct an initial state!  """
            next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
            return next_state


        def setup_step(self, mode='dense'):
            """ Set up dA, dB, dC discretized parameters for stepping """
            self._setup_state()

            # Calculate original C
            dA_L = power(self.L, self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)
            C = _conj(_r2c(self.C)) # (H C N)

            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2),
                C.unsqueeze(-1),
            ).squeeze(-1)
            self.dC = dC

            # Do special preprocessing for different step modes

            self._step_mode = mode
            if mode == 'linear':
                # Linear case: special step function for the state, we need to handle output
                # use conjugate symmetry by default, which affects the output projection
                self.dC = 2*self.dC[:, :, :self.N]
            elif mode == 'diagonal':
                # Eigendecomposition of the A matrix
                L, V = torch.linalg.eig(self.dA)
                V_inv = torch.linalg.inv(V)
                # Check that the eigendedecomposition is correct
                if self.verbose:
                    print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

                # Change the parameterization to diagonalize
                self.dA = L
                self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
                self.dC = contract('h n m, c h n -> c h m', V, self.dC)

            elif mode == 'dense':
                pass
            else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")


        def default_state(self, *batch_shape):
            C = _r2c(self.C)
            N = C.size(-1)
            H = C.size(-2)

            # Cache the tensor contractions we will later do, for efficiency
            # These are put in this function because they depend on the batch size
            if self._step_mode !='linear':
                N *= 2

                if self._step_mode == 'diagonal':
                    self.state_contraction = contract_expression(
                        "h n, ... h n -> ... h n",
                        (H, N),
                        batch_shape + (H, N),
                    )
                else:
                    # Dense (quadratic) case: expand all terms
                    self.state_contraction = contract_expression(
                        "h m n, ... h n -> ... h m",
                        (H, N, N),
                        batch_shape + (H, N),
                    )

                self.input_contraction = contract_expression(
                    "h n, ... h -> ... h n",
                    (H, N), # self.dB.shape
                    batch_shape + (H,),
                )

            self.output_contraction = contract_expression(
                "c h n, ... h n -> ... c h",
                (C.shape[0], H, N), # self.dC.shape
                batch_shape + (H, N),
            )

            state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
            return state

        def step(self, u, state):
            """ Must have called self.setup_step() and created state with self.default_state() before calling this """

            if self._step_mode == 'linear':
                new_state = self._step_state_linear(u, state)
            else:
                new_state = self._step_state(u, state)
            y = self.output_contraction(self.dC, new_state)
            return y, new_state

        def register(self, name, tensor, trainable=False, lr=None, wd=None):
            """Utility method: register a tensor as a buffer or trainable parameter"""

            if trainable:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

            optim = {}
            if trainable and lr is not None:
                optim["lr"] = lr
            if trainable and wd is not None:
                optim["weight_decay"] = wd
            if len(optim) > 0:
                setattr(getattr(self, name), "_optim", optim)


    class HippoSSKernel(nn.Module):
    
        """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

        The SSKernel is expected to support the interface
        forward()
        default_state()
        setup_step()
        step()
        """

        def __init__(
            self,
            H,
            N=64,
            L=1,
            measure="legs",
            rank=1,
            channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
            dt_min=0.001,
            dt_max=0.1,
            trainable=None, # Dictionary of options to train various HiPPO parameters
            lr=None, # Hook to set LR of hippo parameters differently
            length_correction=True, # Multiply by I-A|^L after initialization; can be turned off for initialization speed
            hurwitz=False,
            tie_state=False, # Tie parameters of HiPPO ODE across the H features
            precision=1, # 1 (single) or 2 (double) for the kernel
            resample=False,  # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
            verbose=False,
        ):
            super().__init__()
            self.N = N
            self.H = H
            L = L or 1
            self.precision = precision
            dtype = torch.double if self.precision == 2 else torch.float
            cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
            self.rate = None if resample else 1.0
            self.channels = channels

            # Generate dt
            log_dt = torch.rand(self.H, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

            w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
            C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
            self.kernel = SSKernelNPLR(
                L, w, p, B, C,
                log_dt,
                hurwitz=hurwitz,
                trainable=trainable,
                lr=lr,
                tie_state=tie_state,
                length_correction=length_correction,
                verbose=verbose,
            )

        def forward(self, L=None):
            k, _ = self.kernel(rate=self.rate, L=L)
            return k.float()

        def step(self, u, state, **kwargs):
            u, state = self.kernel.step(u, state, **kwargs)
            return u.float(), state

        def default_state(self, *args, **kwargs):
            return self.kernel.default_state(*args, **kwargs)
        
        
    def get_torch_trans(heads=8, layers=1, channels=64):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
        return nn.TransformerEncoder(encoder_layer, num_layers=layers)

    class S4(nn.Module):

        def __init__(
                self,
                d_model,
                d_state=64,
                l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
                channels=1, # maps 1-dim to C-dim
                bidirectional=False,
                # Arguments for FF
                activation='gelu', # activation in between SS and FF
                postact=None, # activation after FF
                initializer=None, # initializer on FF
                weight_norm=False, # weight normalization on FF
                hyper_act=None, # Use a "hypernetwork" multiplication
                dropout=0.0,
                transposed=True, # axis ordering (B, L, D) or (B, D, L)
                verbose=False,
                # SSM Kernel arguments
                **kernel_args,
            ):


            """
            d_state: the dimension of the state, also denoted by N
            l_max: the maximum sequence length, also denoted by L
            if this is not known at model creation, set l_max=1
            channels: can be interpreted as a number of "heads"
            bidirectional: bidirectional
            dropout: standard dropout argument
            transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

            Other options are all experimental and should not need to be configured
            """


            super().__init__()
            # if verbose:
            #     import src.utils.train
            #     log = src.utils.train.get_logger(__name__)
            #     log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

            self.h = d_model
            self.n = d_state
            self.bidirectional = bidirectional
            self.channels = channels
            self.transposed = transposed

            # optional multiplicative modulation GLU-style
            # https://arxiv.org/abs/2002.05202
            self.hyper = hyper_act is not None
            if self.hyper:
                channels *= 2
                self.hyper_activation = Activation(hyper_act)

            self.D = nn.Parameter(torch.randn(channels, self.h))

            if self.bidirectional:
                channels *= 2


            # SSM Kernel
            self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

            # Pointwise
            self.activation = Activation(activation)
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

            
            # position-wise output transform to mix features
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )
            
            #self.time_transformer = get_torch_trans(heads=8, layers=1, channels=self.h)
            
            
            

        def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
            """
            u: (B H L) if self.transposed else (B L H)
            state: (H N) never needed unless you know what you're doing

            Returns: same shape as u
            """

            if not self.transposed: u = u.transpose(-1, -2)
            L = u.size(-1)
            # Compute SS Kernel
            k = self.kernel(L=L) # (C H L) (B C H L)

            # Convolution
            if self.bidirectional:
                k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
                k = F.pad(k0, (0, L)) \
                        + F.pad(k1.flip(-1), (L, 0)) \

            k_f = torch.fft.rfft(k, n=2*L) # (C H L)
            u_f = torch.fft.rfft(u, n=2*L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
            y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)


            # Compute D term in state space equation - essentially a skip connection
            y = y + contract('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

            # Optional hyper-network multiplication
            if self.hyper:
                y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
                y = self.hyper_activation(yh) * y

            # Reshape to flatten channels
            y = rearrange(y, '... c h l -> ... (c h) l')

            y = self.dropout(self.activation(y))

            if not self.transposed: y = y.transpose(-1, -2)

            y = self.output_linear(y)

            # ysize = b, k, l, requieres l, b, k 
            #y = self.time_transformer(y.permute(2,0,1)).permute(1,2,0)
                
                
            return y, None
            

        def step(self, u, state):
            """ Step one time step as a recurrent model. Intended to be used during validation.

            u: (B H)
            state: (B H N)
            Returns: output (B H), state (B H N)
            """
            assert not self.training

            y, next_state = self.kernel.step(u, state) # (B C H)
            y = y + u.unsqueeze(-2) * self.D
            y = rearrange(y, '... c h -> ... (c h)')
            y = self.activation(y)
            if self.transposed:
                y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
            else:
                y = self.output_linear(y)
            return y, next_state

        def default_state(self, *batch_shape, device=None):
            return self.kernel.default_state(*batch_shape)

        @property
        def d_state(self):
            return self.h * self.n

        @property
        def d_output(self):
            return self.h

        @property
        def state_to_tensor(self):
            return lambda state: rearrange('... h n -> ... (h n)', state)


        
    class S4Layer(nn.Module):
        #S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
        def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
            super().__init__()
            self.s4_layer  = S4(d_model=features, 
                                d_state=N, 
                                l_max=lmax, 
                                bidirectional=bidirectional)
            
            self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity() 
            self.dropout = nn.Dropout2d(dropout) if dropout>0 else nn.Identity()
        
        def forward(self, x):
            #x has shape seq, batch, feature
            x = x.permute((1,2,0)) #batch, feature, seq (as expected from S4 with transposed=True)
            xout, _ = self.s4_layer(x) #batch, feature, seq
            xout = self.dropout(xout)
            xout = xout + x # skip connection   # batch, feature, seq
            xout = xout.permute((2,0,1)) # seq, batch, feature
            return self.norm_layer(xout)
    



    def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
        """
        Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

        Parameters:
        diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                    diffusion steps for batch data
        diffusion_step_embed_dim_in (int, default=128):  
                                    dimensionality of the embedding space for discrete diffusion steps
        
        Returns:
        the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
        """

        assert diffusion_step_embed_dim_in % 2 == 0

        half_dim = diffusion_step_embed_dim_in // 2
        _embed = np.log(10000) / (half_dim - 1)
        _embed = torch.exp(torch.arange(half_dim) * -_embed).to(device)
        _embed = diffusion_steps * _embed
        diffusion_step_embed = torch.cat((torch.sin(_embed),
                                        torch.cos(_embed)), 1)

        return diffusion_step_embed

    def swish(x):
        return x * torch.sigmoid(x)


    class Conv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
            super(Conv, self).__init__()
            self.padding = dilation * (kernel_size - 1) // 2
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
            self.conv = nn.utils.weight_norm(self.conv)
            nn.init.kaiming_normal_(self.conv.weight)

        def forward(self, x):
            out = self.conv(x)
            return out
        
        
    class ZeroConv1d(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(ZeroConv1d, self).__init__()
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
            self.conv.weight.data.zero_()
            self.conv.bias.data.zero_()

        def forward(self, x):
            out = self.conv(x)
            return out


    class Residual_block(nn.Module):
        def __init__(self, res_channels, skip_channels, 
                    diffusion_step_embed_dim_out, in_channels,
                    s4_lmax,
                    s4_d_state,
                    s4_dropout,
                    s4_bidirectional,
                    s4_layernorm):
            super(Residual_block, self).__init__()
            self.res_channels = res_channels


            self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
            
            self.S41 = S4Layer(features=2*self.res_channels, 
                            lmax=s4_lmax,
                            N=s4_d_state,
                            dropout=s4_dropout,
                            bidirectional=s4_bidirectional,
                            layer_norm=s4_layernorm)
    
            self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

            self.S42 = S4Layer(features=2*self.res_channels, 
                            lmax=s4_lmax,
                            N=s4_d_state,
                            dropout=s4_dropout,
                            bidirectional=s4_bidirectional,
                            layer_norm=s4_layernorm)
            
            self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  

            self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
            self.res_conv = nn.utils.weight_norm(self.res_conv)
            nn.init.kaiming_normal_(self.res_conv.weight)

            
            self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
            self.skip_conv = nn.utils.weight_norm(self.skip_conv)
            nn.init.kaiming_normal_(self.skip_conv.weight)

        def forward(self, input_data):
            x, cond, diffusion_step_embed = input_data
            h = x
            B, C, L = x.shape
            assert C == self.res_channels                      
                    
            part_t = self.fc_t(diffusion_step_embed)
            part_t = part_t.view([B, self.res_channels, 1])  
            h = h + part_t
            
            h = self.conv_layer(h)
            h = self.S41(h.permute(2,0,1)).permute(1,2,0)     
            
            assert cond is not None
            cond = self.cond_conv(cond)
            h += cond
            
            h = self.S42(h.permute(2,0,1)).permute(1,2,0)
            
            out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

            res = self.res_conv(out)
            assert x.shape == res.shape
            skip = self.skip_conv(out)

            return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


    class Residual_group(nn.Module):
        def __init__(self, res_channels, skip_channels, num_res_layers, 
                    diffusion_step_embed_dim_in, 
                    diffusion_step_embed_dim_mid,
                    diffusion_step_embed_dim_out,
                    in_channels,
                    s4_lmax,
                    s4_d_state,
                    s4_dropout,
                    s4_bidirectional,
                    s4_layernorm):
            super(Residual_group, self).__init__()
            self.num_res_layers = num_res_layers
            self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

            self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
            self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
            
            self.residual_blocks = nn.ModuleList()
            for n in range(self.num_res_layers):
                self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                        diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                        in_channels=in_channels,
                                                        s4_lmax=s4_lmax,
                                                        s4_d_state=s4_d_state,
                                                        s4_dropout=s4_dropout,
                                                        s4_bidirectional=s4_bidirectional,
                                                        s4_layernorm=s4_layernorm))

                
        def forward(self, input_data):
            noise, conditional, diffusion_steps = input_data

            diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
            # print(diffusion_steps.shape, self.diffusion_step_embed_dim_in, diffusion_step_embed.shape)
            diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
            diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

            h = noise
            skip = 0
            for n in range(self.num_res_layers):
                h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
                skip += skip_n  

            return skip * math.sqrt(1.0 / self.num_res_layers)  


    class SSSDS4Imputer(nn.Module):
        def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                    num_res_layers,
                    diffusion_step_embed_dim_in, 
                    diffusion_step_embed_dim_mid,
                    diffusion_step_embed_dim_out,
                    s4_lmax,
                    s4_d_state,
                    s4_dropout,
                    s4_bidirectional,
                    s4_layernorm):
            super(SSSDS4Imputer, self).__init__()

            self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
            
            self.residual_layer = Residual_group(res_channels=res_channels, 
                                                skip_channels=skip_channels, 
                                                num_res_layers=num_res_layers, 
                                                diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                                diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                in_channels=in_channels,
                                                s4_lmax=s4_lmax,
                                                s4_d_state=s4_d_state,
                                                s4_dropout=s4_dropout,
                                                s4_bidirectional=s4_bidirectional,
                                                s4_layernorm=s4_layernorm)
            
            self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                            nn.ReLU(),
                                            ZeroConv1d(skip_channels, out_channels))

        def forward(self, input_data):
            
            noise, conditional, mask, diffusion_steps = input_data 

            conditional = conditional * mask
            conditional = torch.cat([conditional, mask.float()], dim=1)

            x = noise
            x = self.init_conv(x)
            x = self.residual_layer((x, conditional, diffusion_steps))
            y = self.final_conv(x)

            return y


    def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-4, end=2e-2):
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

    # expect=condition_expect(dataset_x)
    diff_total_steps=config_diffusion['diff_total_steps']
    betas=make_beta_schedule(schedule=config_diffusion['noise_schedule'], 
                            num_timesteps=diff_total_steps,
                            start=config_diffusion['noise_start'], 
                            end=config_diffusion['noise_end']).to(device)
    alphas=1-betas
    alphas_bar=torch.cumprod(alphas,0).to(device)
    alphas_bar_sqrt=torch.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt=torch.sqrt(1-alphas_bar)


    def ddpm_model_forward(model,
                        future,
                        noisy_data,
                        diff_step,
                        history,
                        x_t_mark,
                        history_mark,
                        mask,):
        B,T_his,D = history.shape
        B,T_fut,D = future.shape
        conditional = torch.cat([history, 0*future],dim=1)
        diff_step = diff_step.view(B,-1)
        input_data = (noisy_data.permute(0, 2, 1), conditional.permute(0, 2, 1), mask.permute(0, 2, 1), diff_step)
        # 
        model_output = model(input_data)

        return model_output.permute(0, 2, 1)

    def extract(input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)


    def ddpm_loss_fn(model,
                    batch_x,
                    batch_history,
                    batch_x_mark,
                    batch_history_mark,
                    mask_cond_prob=0.75,
                    use_fft=config_diffusion['use_fft'],
                    diff_total_steps=diff_total_steps,
                    alphas_bar_sqrt=alphas_bar_sqrt,
                    one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
                    device=device):
        model.train()
        B, T_his, D = batch_history.shape
        B, T_fut, D = batch_x.shape

        cat_data = torch.cat([batch_history, batch_x], dim =1)
        
        # ✅  diffusion step t (！)
        diff_step = torch.randint(0, diff_total_steps, size=(B,1)).to(device)


        # mask
        cond_masks = (torch.rand(B, T_his + T_fut, D) < mask_cond_prob).to(device)  # shape [B, seq_len]
        cond_masks = cond_masks.float()
        # cond_masks = cond_masks.unsqueeze(-1)  # shape [B, seq_len, 1]

        # ✅  alpha_bar_sqrt  one_minus_alphas_bar_sqrt（ sigma）
        alpha_bar_sqrt_batch = extract(alphas_bar_sqrt, diff_step.view(-1), cat_data)
        one_minus_alphas_bar_sqrt_batch = extract(one_minus_alphas_bar_sqrt, diff_step.view(-1), cat_data)

        #  x_t
        noise = torch.randn_like(cat_data)
        noisy_data = alpha_bar_sqrt_batch * cat_data + one_minus_alphas_bar_sqrt_batch * noise

        # 
        out = ddpm_model_forward(model=model,
                                future = batch_x,
                                noisy_data = noisy_data,
                                diff_step = diff_step,
                                history = batch_history,
                                x_t_mark = None,
                                history_mark = None,
                                mask = cond_masks,)
        # print(out.shape,cat_data.shape)

        # 
        l2_loss = (cat_data - out).square().mean()
        if use_fft:
            fft1 = torch.fft.fft(out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(cat_data.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = (torch.real(fft1) - torch.real(fft2)).square().mean() \
                        + (torch.imag(fft1) - torch.imag(fft2)).square().mean()
            fourier_loss = use_fft * fourier_loss
        else:
            fourier_loss = 0
            
        return l2_loss + fourier_loss
        


    set_seed(514 + seed_idx)
    in_channels = num_features          
    out_channels = num_features          
    res_channels = 64         #  S4 
    skip_channels = 64
    num_res_layers = 6        # 
    diffusion_step_embed_dim_in = 128
    diffusion_step_embed_dim_mid = 512
    diffusion_step_embed_dim_out = 512
    s4_lmax = config_diffusion['pred_len'] + config_diffusion['windows']      
    s4_d_state = 64      # 64
    s4_dropout = 0.1      # 0.1
    s4_bidirectional = True                     # 
    s4_layernorm = True                         # 



    model_diffusion = SSSDS4Imputer(
        in_channels=in_channels,
        res_channels=res_channels,
        skip_channels=skip_channels,
        out_channels=out_channels,
        num_res_layers=num_res_layers,
        diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
        s4_lmax=s4_lmax,
        s4_d_state=s4_d_state,
        s4_dropout=s4_dropout,
        s4_bidirectional=s4_bidirectional,
        s4_layernorm=s4_layernorm
    ).to(device)
    print(sum(p.numel() for p in model_diffusion.parameters()))
    optimizer = torch.optim.AdamW(
        model_diffusion.parameters(),
        lr=config_diffusion['lr'],
        weight_decay=config_diffusion['weight_decay'] 
    )


    #  import torch ， multiprocessing 
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    #  torch
    import torch

    val_loss = []
    num_training_steps = config_diffusion['num_training_steps']

    # 
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


            batch_history = batch_x.to(device).float()
            # batch_future = batch_y.to(device).float()
            batch_future= future_cw_train[i][1].to(device).float()
            batch_x_t_mark = batch_y_mark.to(device).float()
            batch_history_mark = batch_x_mark.to(device).float()

            loss = ddpm_loss_fn(model = model_diffusion, 
                                batch_x = batch_future, 
                                batch_history = batch_history, 
                                batch_x_mark=batch_x_t_mark,
                                batch_history_mark=batch_history_mark,
                                mask_cond_prob=1,
                                diff_total_steps = diff_total_steps,
                                alphas_bar_sqrt=alphas_bar_sqrt,
                                one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
                                device=device)
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
                # batch_future = batch_y.to(device).float()
                batch_future= future_cw_val[i][1].to(device).float()
                batch_x_t_mark = batch_y_mark.to(device).float()
                batch_history_mark = batch_x_mark.to(device).float()

                loss = ddpm_loss_fn(model = model_diffusion, 
                                    batch_x = batch_future, 
                                    batch_history = batch_history, 
                                    batch_x_mark=batch_x_t_mark,
                                    batch_history_mark=batch_history_mark,
                                    mask_cond_prob=1,
                                    diff_total_steps = diff_total_steps,
                                    alphas_bar_sqrt=alphas_bar_sqrt,
                                    one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
                                    device=device)
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
                batch_future= future_cw_test[i][1].to(device).float()
                batch_x_t_mark = batch_y_mark.to(device).float()
                batch_history_mark = batch_x_mark.to(device).float()
                
                test_loss = ddpm_loss_fn(model = model_diffusion, 
                                        batch_x = batch_future, 
                                        batch_history = batch_history, 
                                        batch_x_mark=batch_x_t_mark,
                                        batch_history_mark=batch_history_mark,
                                        mask_cond_prob=1,
                                        diff_total_steps = diff_total_steps,
                                        alphas_bar_sqrt=alphas_bar_sqrt,
                                        one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
                                        device=device)

                test_total += test_loss.item()
                test_avg = test_total / len(dataloader_cw.test_loader)
            
            

            # 
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_step = step
                best_model_state = {k: v.clone() for k, v in model_diffusion.state_dict().items()}

        # print(f"Step {step}: Train Loss = {total_loss:.4f} | Val Loss = {val_avg:.4f} | Test Loss = {test_avg:.4f}")


    plt.plot(val_loss[:])


    # ==========  ==========
    if best_model_state is not None:
        model_diffusion.load_state_dict(best_model_state)
        print(f"\nBest Val Loss = {best_val_loss:.4f} at Step {best_step}")
    else:
        print("No valid model state was saved.")


    def ddim_sampler(net, 
                    latents, 
                    history,
                    x_t_mark,
                    history_mark,
                    alphas_bar_sqrt,
                    one_minus_alphas_bar_sqrt,
                    pred_len = config_diffusion['pred_len'],
                    timesteps=None,
                    num_steps=25,
                    diff_total_steps=config_diffusion['diff_total_steps'],
                    eta=0.0,  # η=0 ，η>0 
                    device=device):
        
        net.eval()
        traj = []
        future_zero =  torch.zeros(latents.shape[0],pred_len,latents.shape[2]).to(device)

        cond_mask = torch.cat([torch.ones_like(history).to(device), future_zero],dim=1)

        #  latent 
        x_t = latents.to(torch.float32).to(device)
        traj.append(x_t)

        # ： 0~999  num_steps （）
        if timesteps is None:
            timesteps = torch.linspace(diff_total_steps-1, 0, num_steps, dtype=torch.long, device=device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            t_tensor = torch.full((x_t.shape[0],), t.item(), dtype=torch.long, device=device)
            t_next_tensor = torch.full((x_t.shape[0],), t_next.item(), dtype=torch.long, device=device)
            x0_pred = ddpm_model_forward(model= net,
                                        future =future_zero,
                                        noisy_data= x_t,
                                        diff_step = t_tensor,
                                        history = history,
                                        x_t_mark = None,
                                        history_mark = None,
                                        mask = cond_mask,)
            
            #  alpha_bar
            a_t = alphas_bar_sqrt[t]
            a_t_next = alphas_bar_sqrt[t_next]
            om_a_t = one_minus_alphas_bar_sqrt[t]
            om_a_t_next = one_minus_alphas_bar_sqrt[t_next]

            # 1.  x_0
            noise_pred = (x_t - a_t * x0_pred) / om_a_t

            # 2.  θ_{t-1}（）
            sigma_t = eta * torch.sqrt((om_a_t_next ** 2) / (om_a_t ** 2) * (1 - (a_t ** 2) / (a_t_next ** 2)))
            noise = torch.randn_like(x_t) if eta > 0 else 0.0

            x_t = a_t_next * x0_pred + om_a_t_next * noise_pred + sigma_t * noise
            traj.append(x_t)

        #  timestep = 0
        t_final = timesteps[-1]
        t_tensor = torch.full((x_t.shape[0],), t_final.item(), dtype=torch.long, device=device)
        x0_pred = ddpm_model_forward(model= net,
                                    future =future_zero,
                                    noisy_data= x_t,
                                    diff_step = t_tensor,
                                    history = history,
                                    x_t_mark = None,
                                    history_mark = None,
                                    mask = cond_mask,)
        

        traj.append(x0_pred)

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


    for eta in [0, 0.25, 0.5]:
        generated_batch = []
        # eta = 0.
        num_steps = 50
        use_cond = True
        num = 100


        set_seed(1920)
        from tqdm import tqdm
        sample_device = device
        with torch.inference_mode():
            for iii in tqdm(range(num)):
                latents = torch.randn(batch_history.shape[0], config_diffusion['pred_len'] + config_diffusion['windows'], num_features).to(sample_device)
                if use_cond:
                    samples = ddim_sampler(net = model_diffusion.to(sample_device), 
                                            latents = latents.to(sample_device), 
                                            history = batch_history.to(sample_device),
                                            # history = None,
                                            x_t_mark = batch_x_t_mark.to(sample_device),
                                            history_mark = batch_history_mark.to(sample_device),
                                            alphas_bar_sqrt = alphas_bar_sqrt.to(sample_device),
                                            one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(sample_device),
                                            timesteps=None,
                                            num_steps=num_steps,
                                            eta=eta,  # η=0 ，η>0 
                                            device=sample_device)

                generated_batch.append(samples[-1].unsqueeze(-1))


        generated_batch[iii].shape


        cov_sqrt_test = future_cw_test[0][3]
        miu_test = future_cw_test[0][4]
        for iii in range(len(generated_batch)):
            generated_batch[iii] = (torch.einsum('btij,btj->bti', cov_sqrt_test, generated_batch[iii].squeeze(-1)[:,168:,:]) +  miu_test).unsqueeze(-1)  # [B,T,D]



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
        set_seed(810 + seed_idx)
        context_fid = Context_FID(ori_data = torch.cat([batch_history_original,batch_future_original],dim=1).detach().cpu().numpy(), 
                    test_data = torch.cat([batch_history_original,batch_future_original],dim=1).detach().cpu().numpy(), 
                    generated_data = pred_xy.detach().cpu().numpy())


        cacf_list = []
        for iii in range(len(generated_batch)):
            cacf_list.append((cacf_torch(x = torch.cat([batch_history_original,batch_future_original],dim=1)) \
                            -cacf_torch(x =  torch.cat([batch_history_original,generated_batch[iii].squeeze(-1)],dim=1))).abs().mean())
        cacf_mean, cacf_std = torch.tensor(cacf_list).mean(),torch.tensor(cacf_list).std()


        with open(log_file, "a", encoding="utf-8") as f:
            record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed_idx": seed_idx,
                "eta": eta,
                "metrics": metrics_dict,
                "Context_FID": float(context_fid),
                "cacf_mean": float(cacf_mean),
                "cacf_std": float(cacf_std)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


