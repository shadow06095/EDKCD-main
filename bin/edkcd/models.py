import torch
import torch.nn as nn
try:
    from efficient_kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False


class MLPHankelEncoder(nn.Module):
    """MLP-based Hankel encoder: sliding window -> shared MLP per channel."""
    def __init__(self, embed_dim, out_channels, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_channels))
        self.net = nn.Sequential(*layers)
        self.pad = nn.ConstantPad1d((embed_dim - 1, 0), 0.0)

    def forward(self, x):
        BN, _, T = x.shape
        x_pad = self.pad(x)
        x_unfold = x_pad.unfold(2, self.embed_dim, 1)
        x_windows = x_unfold.squeeze(1)
        flat = x_windows.reshape(BN * T, self.embed_dim)
        out = self.net(flat)
        out = out.reshape(BN, T, self.out_channels)
        out = out.transpose(1, 2)
        return out


class KANHankelEncoder(nn.Module):
    """KAN-based Hankel encoder."""
    def __init__(self, embed_dim, out_channels, kan_layers=None,
                 grid_size=5, spline_order=3):
        super().__init__()
        if not KAN_AVAILABLE:
            raise ImportError("KAN encoder requires efficient_kan")
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        if kan_layers is None:
            layers_hidden = [embed_dim, 8, out_channels]
        else:
            layers_hidden = [embed_dim] + list(kan_layers) + [out_channels]
        self.kan = KAN(layers_hidden=layers_hidden,
                       grid_size=grid_size, spline_order=spline_order)
        self.pad = nn.ConstantPad1d((embed_dim - 1, 0), 0.0)

    def forward(self, x):
        BN, _, T = x.shape
        x_pad = self.pad(x)
        x_unfold = x_pad.unfold(2, self.embed_dim, 1)
        x_windows = x_unfold.squeeze(1)
        out = self.kan(x_windows)
        out = out.transpose(1, 2)
        return out


class VARP(nn.Module):
    """Vector Auto-Regressive with Pooling: encoder-fusion-decoder architecture."""
    def __init__(self, c_in, lag=1, encoder_layers=None, decoder_layers=None,
                 seed=0, encoder_type='mlp',
                 kan_embed_dim=5, kan_hidden=None,
                 channel_independent=False, mlp_dropout=0.1):
        super().__init__()
        if encoder_layers is None:
            encoder_layers = 8 * [12]
        if decoder_layers is None:
            decoder_layers = 8 * [12]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.c_in = c_in
        self.lag = lag
        self.const_pad = nn.ConstantPad1d((lag - 1, 0), 0)
        self.scale_factor = 1
        self.encoder_type = encoder_type
        self.decoder_mode = encoder_type

        # encoder
        if encoder_type == 'tcn':
            raise ValueError("TCN encoder is no longer supported. Use 'mlp' or 'kan'.")
        elif encoder_type == 'kan':
            if kan_hidden is None:
                kan_hidden = [kan_embed_dim, 8, encoder_layers[-1]]
            self.encoder = KANHankelEncoder(embed_dim=kan_embed_dim,
                                            out_channels=encoder_layers[-1],
                                            kan_layers=kan_hidden)
            self.n_channels = encoder_layers[-1]
            self.kan_hidden = kan_hidden
        elif encoder_type == 'mlp':
            mlp_hidden = [32, 32] if kan_hidden is None else list(kan_hidden)
            self.encoder = MLPHankelEncoder(embed_dim=kan_embed_dim,
                                            out_channels=encoder_layers[-1],
                                            hidden_dims=mlp_hidden)
            self.n_channels = encoder_layers[-1]
            self.mlp_hidden = mlp_hidden
        else:
            raise ValueError("encoder_type must be 'mlp' or 'kan'")

        self.identity_pool = nn.AvgPool1d(1)

        # decoder
        if self.decoder_mode == 'tcn':
            raise ValueError("TCN decoder is no longer supported. Use 'mlp' or 'kan'.")
        elif self.decoder_mode == 'kan':
            if not KAN_AVAILABLE:
                raise ImportError("KAN decoder requires efficient_kan")
            layers_hidden = [self.n_channels] + list(reversed(self.kan_hidden)) + [1]
            self.kan_decoder = KAN(layers_hidden=layers_hidden)
        elif self.decoder_mode == 'mlp':
            hidden = list(reversed(self.mlp_hidden))
            layers = []
            in_dim = self.n_channels
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(mlp_dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.mlp_decoder = nn.Sequential(*layers)

        C = self.n_channels
        if channel_independent:
            self.K = nn.Parameter(torch.zeros(c_in, c_in, C, lag).normal_(0, 0.01))
        else:
            self.K = nn.Parameter(torch.zeros(c_in, c_in, C, C, lag).normal_(0, 0.01))
        self.channel_independent = channel_independent

    def _decode(self, x):
        if self.decoder_mode == 'tcn':
            raise ValueError("TCN decoder is no longer supported.")
        elif self.decoder_mode == 'kan':
            BN, C, T = x.shape
            x = x.transpose(1, 2).reshape(BN * T, C)
            x = self.kan_decoder(x)
            x = x.reshape(BN, T, 1).transpose(1, 2)
        elif self.decoder_mode == 'mlp':
            BN, C, T = x.shape
            x = x.transpose(1, 2).reshape(BN * T, C)
            x = self.mlp_decoder(x)
            x = x.reshape(BN, T, 1).transpose(1, 2)
        return x

    def forward(self, x, var_fusion_enabled):
        B, N, T = x.shape
        C = self.n_channels
        x_enc = self.encoder(x.reshape(B * N, 1, T))
        x_enc = self.identity_pool(x_enc)

        if var_fusion_enabled:
            x_pad = self.const_pad(x_enc)
            x_unfold = x_pad.unfold(2, self.lag, 1)
            x_reshaped = x_unfold.reshape(B, N, C, -1, self.lag)
            x_perm = x_reshaped.permute(0, 1, 3, 2, 4)
            if self.channel_independent:
                x_fused = torch.einsum('bitcl,ijcl->bjtc', x_perm, self.K)
                x_fused = x_fused.permute(0, 1, 3, 2).reshape(B * N, C, -1)
            else:
                x_fused = torch.einsum('bitcl,ijcdl->bjtd', x_perm, self.K)
                x_fused = x_fused.reshape(B * N, x_fused.shape[2], C)
                x_fused = x_fused.transpose(1, 2)
            x_out = self._decode(x_fused)
        else:
            x_out = self._decode(x_enc)
        x_out = x_out.reshape(B, N, -1)
        return x_out

    def compute_latent_fusion(self, x_enc):
        B_merged = x_enc.size(0) // self.c_in
        x_pad = self.const_pad(x_enc)
        x_unfold = x_pad.unfold(2, self.lag, 1)
        x_reshaped = x_unfold.reshape(B_merged, self.c_in, self.n_channels, -1, self.lag)
        x_perm = x_reshaped.permute(0, 1, 3, 2, 4)
        if self.channel_independent:
            x_fused = torch.einsum('bitcl,ijcl->bjtc', x_perm, self.K)
            x_fused = x_fused.permute(0, 1, 3, 2).reshape(B_merged * self.c_in, -1, self.n_channels)
            x_fused = x_fused.transpose(1, 2)
        else:
            x_fused = torch.einsum('bitcl,ijcdl->bjtd', x_perm, self.K)
            x_fused = x_fused.reshape(B_merged * self.c_in, -1, x_fused.shape[3])
            x_fused = x_fused.transpose(1, 2)
        return x_fused

    def get_regularized_params(self, stage=1):
        if stage == 0:
            if self.decoder_mode == 'kan':
                return list(self.kan_decoder.parameters())
            else:
                return list(self.mlp_decoder.parameters())
        else:
            if self.decoder_mode == 'kan':
                return [self.K] + list(self.kan_decoder.parameters())
            else:
                return [self.K] + list(self.mlp_decoder.parameters())

    def get_trainable_params(self):
        return [self.K]