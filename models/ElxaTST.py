import copy
import math
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.multi_scale    = configs.multi_scale

        self.use_norm = configs.use_norm

        # Model config
        self.kernel_size = configs.kernel_size
        self.feature_size = configs.enc_in

        # Patching
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = (self.seq_len - self.patch_len + self.stride) // self.stride + 1
        self.dec_dim = int(self.patch_num * (configs.d_model / self.patch_len))

        # Dformer_Line
        self.percent_dx = configs.percent_dx
        self.percent_dy = configs.percent_dy
        self.n_dx = math.ceil(self.percent_dx * self.patch_num)
        self.n_dy = math.ceil(self.percent_dy * configs.enc_in)

        self.moving_avg = MovingAvg(self.kernel_size, stride=1)

        # Data Embedding
        self.trend_embedding = PatchEmbedding(
            configs.d_model,
            configs.patch_len,
            configs.stride,
            configs.dropout,
            "linear",
            "trend",
            configs.embed,
            configs.freq,
        )
        self.season_embedding = PatchEmbedding(
            configs.d_model,
            configs.patch_len,
            configs.stride,
            configs.dropout,
            "linear",
            "season",
            configs.embed,
            configs.freq,
        )

        # STL
        self.map_trend = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_season = nn.Sequential(
            nn.Linear(configs.seq_len, 4 * configs.seq_len),
            nn.ReLU(),
            nn.Linear(4 * configs.seq_len, configs.seq_len),
        )
        self.top_sk = configs.top_sk
        self.top_dk = configs.top_dk
        # Encoder
        trend_encoder_layer = DeformableLineTrEncoderLayer(
            d_model=configs.d_model,
            d_ffn=configs.d_ff,
            dropout=configs.dropout,
            activation="relu",
            n_heads=8,
            n_dx=self.n_dx,
            n_dy=self.n_dy,
            n_variates=self.feature_size,
            interpolation=None,
        )
        season_encoder_layer = DeformableLineTrEncoderLayer(
            d_model=configs.d_model,
            d_ffn=configs.d_ff,
            dropout=configs.dropout,
            activation="relu",
            n_heads=8,
            n_dx=self.n_dx,
            n_dy=self.n_dy,
            n_variates=self.feature_size,
            interpolation=None,
        )
        self.trend_encoder = DeformableLineTrEncoder(
            trend_encoder_layer,
            num_layers=configs.e_layers,
            embed_dims=configs.d_model,
        )
        self.season_encoder = DeformableLineTrEncoder(
            season_encoder_layer,
            num_layers=configs.e_layers,
            embed_dims=configs.d_model,
        )
        # Compress
        self.out_trend_conv_layer = nn.Conv2d(
            configs.d_model,
            configs.d_model // self.patch_len,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        self.out_season_conv_layer = nn.Conv2d(
            configs.d_model,
            configs.d_model // self.patch_len,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        # Decoder
        self.trend_projector = nn.Linear(self.dec_dim, self.pred_len, bias=True)
        self.season_projector = nn.Linear(self.dec_dim, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        :params x_enc       (bs, seq_len, n_var)    : origin data
        :params x_mark_enc  (bs, seq_len, 4)        : time information

        :returns            ()

        """

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        trend_local = self.moving_avg(x_enc)
        season_local = x_enc - trend_local
        trend_local = x_enc

        trend_local = self.map_trend(trend_local.transpose(1, 2)).transpose(1, 2) + trend_local
        season_local = self.map_season(season_local.transpose(1, 2)).transpose(1, 2) + season_local

        emb_trend = self.trend_embedding(trend_local.to(torch.bfloat16))  # [bs, n_var, n_patch, dim]
        emb_season = self.season_embedding(season_local.to(torch.bfloat16))  # [bs, n_var, n_patch, dim]
        # Trend Encoder
        bs, h, w, c = emb_trend.shape
        spatial_shape = (h, w)
        trend_src = emb_trend.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)  # [bs, n_var*n_patch, dim]
        trend_enc_out = self.trend_encoder(trend_src, spatial_shape, self.top_sk, self.top_dk)
        trend_enc_out = trend_enc_out.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
        trend_enc_out = self.out_trend_conv_layer(trend_enc_out).permute(0, 2, 3, 1).flatten(2)
        trend_dec_out = self.trend_projector(trend_enc_out).permute(0, 2, 1)

        # Season Encoder
        season_src = emb_season.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)  # [bs, n_var*n_patch, dim]
        season_enc_out = self.season_encoder(season_src, spatial_shape, self.top_sk, self.top_dk)
        season_enc_out = season_enc_out.reshape(bs, h, w, -1).reshape(bs, h, w, -1).permute(0, 3, 1, 2)
        season_enc_out = self.out_season_conv_layer(season_enc_out).permute(0, 2, 3, 1).flatten(2)
        season_dec_out = self.season_projector(season_enc_out).permute(0, 2, 1)

        # dec_out = trend_dec_out + season_dec_out + noise_dec_out
        dec_out = trend_dec_out + season_dec_out

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out  # check ???


class DeformableLineTrEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        n_dx=4,
        n_dy=4,
        n_variates=1,
        interpolation="unilinear",
    ):
        super().__init__()

        # self attention
        self.self_attn = DeformableLineAttention(d_model, n_heads, n_dx, n_dy, n_variates, interpolation)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_function(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shape, xy_coord, top_sk, top_dk):
        src = self.with_pos_embed(src, pos)
        src2 = self.self_attn(src, reference_points, src, spatial_shape, xy_coord, top_sk, top_dk)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)
        return src


class DeformableLineTrEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, embed_dims):
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(encoder_layer, num_layers)

        self.data_to_xy_space = nn.Linear(embed_dims, 2)

    def get_reference_points(self, spatial_shape, device):
        h, w = spatial_shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / h
        ref_x = ref_x.reshape(-1)[None] / w
        reference_points = torch.stack((ref_x, ref_y), -1)

        return reference_points

    def forward(self, src, spatial_shape, top_sk, top_dk, pos=None):
        output = src
        reference_points = self.get_reference_points(spatial_shape, src.device)  # [1, n_var*n_patch, 2]

        # tv - xy mapping
        xy_coord = self.data_to_xy_space(src)

        for i, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shape, xy_coord, top_sk, top_dk)
        return output


class DeformableLineAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_dx=4, n_dy=4, n_variates=1, interpolation="unilinear"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_dx = n_dx
        self.n_dy = n_dy
        self.interpolation = interpolation
        self.n_variates = n_variates
        self.sampling_dx = nn.Linear(in_features=d_model, out_features=n_dy)
        self.channel_patch_embedding = nn.Linear(d_model, 16)

        self.attn = AttentionLayer(FullAttention, d_model, n_heads)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.sampling_dx.weight.data)
        constant_(self.sampling_dx.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shape, xy_coord, top_sk, top_dk):
        N, Len_q, q_dim = query.shape
        N, Len_in, in_dim = input_flatten.shape
        h, w = input_spatial_shape

        n_patches = Len_q // h

        query_ = query.view(N, h, -1, q_dim)
        summarized_query = torch.sum(query_, dim=2).view(N, h, q_dim) / n_patches  # [B, N, D]
        sq_emb = self.channel_patch_embedding(summarized_query)
        cosine_similarities = F.cosine_similarity(sq_emb.unsqueeze(1), sq_emb.unsqueeze(2), dim=3)  # [B, N, N]
        _topk_similarities, topk_indices = torch.topk(cosine_similarities, self.n_dy, dim=2)  # [B, N, n_dy]

        sampling_x = self.sampling_dx(query_)  # [B, N, n_patch, n_dy], 0 to 1
        sampling_y = topk_indices.float().unsqueeze(2)  # [B, N, 1, n_dy], 0 to 1
        sampling_y = sampling_y.repeat(1, 1, n_patches, 1)  # [B, N, n_patch, n_dy],

        zero_dx = torch.zeros((N, h, w, 1)).to(sampling_x.device)
        zero_dy = torch.zeros((N, h, w, 1)).to(sampling_y.device)

        sampling_dx = torch.cat((sampling_x, zero_dx), dim=-1)  # [B, N, n_patch, n_dy+1]
        sampling_dy = torch.cat((sampling_y, zero_dy), dim=-1)  # [B, N, n_patch, n_dy+1]

        sampling_x = reference_points[:, :, 0].repeat(N, 1).view(N, h, w, -1) + sampling_dx / w  # [B, N, n_patch, n_dy+1]
        sampling_y = reference_points[:, :, 1].repeat(N, 1).view(N, h, w, -1) + sampling_dy / h  # [B, N, n_patch, n_dy+1]
        # print("sampling_x", sampling_x.shape)

        sampling_coords = torch.cat((sampling_x, sampling_y), dim=-1).view(N, h, -1, 2)  # [B, N, n_patch*n_dy, 2]

        value = input_flatten.view(N, h, w, self.d_model).repeat(
            1,
            1,
            self.n_dy + 1,
            1,
        )  # [bs, n_var, n_patch*n_dy, h_dim]
        sampling_coords = sampling_coords * 2 - 0.99999  # For using F.grid_sample

        sampling_value = F.grid_sample(
            value.permute(0, 3, 1, 2),
            sampling_coords,  # [bs, n_var, n_patch, 2]
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        query = query.unsqueeze(-2)
        sampling_value = sampling_value.repeat(1, n_patches, 1, 1)

        out, attn = self.attn(queries=query, keys=sampling_value, values=sampling_value, attn_mask=None)
        out = out.reshape(N, Len_q, -1)
        return out


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, window_size, stride):
        super(MovingAvg, self).__init__()
        self.window_size = window_size
        self.avg = nn.AvgPool1d(kernel_size=self.window_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.window_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.window_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


def _get_clones(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])


def _get_activation_function(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu

    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention(mask_flag=False)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # B, L, _ = queries.shape
        # _, S, _ = keys.shape
        bs, n_q, one, dim = queries.shape
        bs, n_q, n_sample, dim = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(bs, n_q, one, H, -1)
        keys = self.key_projection(keys).view(bs, n_q, n_sample, H, -1)
        values = self.value_projection(values).view(bs, n_q, n_sample, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(bs, n_q, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = True
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, n_q, one, H, E = queries.shape
        B, n_q, n_sample, H, E = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("bklhe,bkshe->bkhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, n_q, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bkhls,bkshd->bklhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, val_embedding_type, flag, embed_type="fixed", freq="h"):
        """
        :params d_model             :
        :params patch_len           :
        :params stride              :
        :params dropout             :
        :params multi_scale         :
        :params avgpool             :
        :params val_embedding_type  :
        :params embed_type          :
        :params freq                :
        """

        super(PatchEmbedding, self).__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Embedding
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.val_embedding_type = val_embedding_type
        self.embed_type = embed_type
        self.freq = freq
        self.token_embedding = TokenEmbedding(patch_len, d_model, val_embedding_type, flag)
        self.temporal_PE = TemporalEmbedding(c_in=patch_len, d_model=d_model)
        self.var_PE = PositionalEmbedding(d_model)
        self.sin_PE = SinusoidalPositionalEmbedding(d_model // 2)

    def forward(self, x, x_mark=None):
        """
        :params x       : [bs, seq_len, n_var]
        :params x_mark  : [bs, seq_len, time_attr]
        """

        x = x.permute(0, 2, 1)
        bs = x.size(0)

        x = self.padding_patch_layer(x)

        x_feat_PE = self.var_PE(x).repeat(bs, 1, 1).unsqueeze(-2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.token_embedding(x)

        if x_mark is not None:
            x_mark = x_mark.permute(0, 2, 1)
            x_mark = self.padding_patch_layer(x_mark)
            x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x_mark = self.temporal_PE(x_mark.to(torch.float32))
            x = self.dropout(x + x_feat_PE + x_mark)
        else:
            x_PE = self.sin_PE(x).to(x.device)
            x = self.dropout(x + x_PE)

        return x


class ReplicationPad1d(nn.Module):
    def __init__(self, padding):
        """
        :params padding :
        """
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input):
        # zero_padding = torch.zeros(input.size(0), input.size(1), self.padding[-1], device=input.device, dtype=input.dtype)
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output  # torch.Size([8, 1, 520])


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, val_embedding_type="linear", flag="trend"):
        """
        :params c_in    :
        :params d_model :
        :params val_embedding_type  :
        """
        super(TokenEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.val_embedding_type = val_embedding_type

        if val_embedding_type == "linear":
            self.linear_layer = nn.Linear(in_features=c_in, out_features=d_model, bias=False)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

        else:
            raise RuntimeError("You must select 'linear")

    def forward(self, x):
        """
        :params x   :
        """
        x = x.to(torch.float32)
        bs, n_var, n_seq, patch_len = x.shape

        x = x.reshape(-1, patch_len)
        if self.val_embedding_type == "linear":
            # x = self.linear(x)
            x = self.linear_layer(x).reshape(bs, n_var, n_seq, -1)

        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TemporalEmbedding, self).__init__()
        self.embed = nn.Linear(c_in, d_model)

    def forward(self, x):
        """
        :params x   : [bs, n_var, n_seq, patch_len]
        """
        bs, n_var, n_seq, patch_len = x.shape
        x = x.reshape(-1, patch_len)
        x = self.embed(x).reshape(bs, n_var, n_seq, -1)
        x = torch.sum(x, dim=1, keepdim=True)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, data):
        bs, y, x, emb = data.shape
        data_ones = torch.ones((bs, y, x))
        y_embed = data_ones.cumsum(1, dtype=torch.float32)
        x_embed = data_ones.cumsum(2, dtype=torch.float32)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=data_ones.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)

        return pos
