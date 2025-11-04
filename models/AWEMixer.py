import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt # 恢复使用原始的小波变换库

# =====================================================================================
# 1. 核心模块 (RevIN, MixerBlock CoherentGatingBlock)
# =====================================================================================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10) # 增加稳定性
        x = x * self.stdev + self.mean
        return x

class CoherentGatingBlock(nn.Module):
    """相干门控融合块 (与原版基本一致)"""
    def __init__(self, d_model, n_heads, dropout):
        super(CoherentGatingBlock, self).__init__()
        self.context_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # 优化门控，使其更具表达能力
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_t, h_w_pyramid):
        # h_t: (B*C, 1, D), h_w_pyramid: (B*C, Num_Bands, D)
        # 1. h_t 查询整个小波金字塔，合成上下文
        context, _ = self.context_attention(query=h_t, key=h_w_pyramid, value=h_w_pyramid)
        h_t_enhanced = self.norm1(h_t + context)
        
        # 2. 门控融合
        # 使用 unsqueeze 保持维度一致性，避免 squeeze-unsqueeze
        g = self.gate(torch.cat([h_t, h_t_enhanced], dim=-1))
        
        h_t_fused = g * h_t_enhanced + (1 - g) * h_t
        return self.norm2(h_t_fused)

class MixerBlock(nn.Module):
    """跨尺度混合器"""
    def __init__(self, num_scales, d_model, tokens_mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_scales, tokens_mlp_dim), nn.GELU(),
            nn.Dropout(dropout), # 增加Dropout
            nn.Linear(tokens_mlp_dim, num_scales)
        )

    def forward(self, x):
        # x: (B*C, S, D)
        residual = x
        x = self.norm(x).transpose(1, 2) # (B*C, D, S)
        x = self.token_mixing(x).transpose(1, 2) # (B*C, S, D)
        return residual + x

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2.  频段路由器 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FrequencyRouter(nn.Module):
    """
    频段路由器: 根据输入信号的周期性特征，动态为每个小波频段生成权重。
    借鉴Pathformer的思想，但应用于频域。
    """
    def __init__(self, seq_len, num_wavelet_bands, top_k=4, d_model=128):
        super().__init__()
        self.top_k = top_k
        self.seq_len = seq_len
        self.num_wavelet_bands = num_wavelet_bands

        # 用于提取周期性特征的线性层
        self.period_extractor = nn.Linear(seq_len // 2 + 1, d_model)
        # 生成路由权重的最终线性层
        self.router_generator = nn.Linear(d_model, num_wavelet_bands)

    def forward(self, x_ci):
        # x_ci: (B*C, L_in)
        # 1. 使用FFT提取信号的周期性特征
        fft_result = torch.fft.rfft(x_ci, n=self.seq_len, dim=-1)
        # 取振幅作为特征，形状为 (B*C, L_in//2 + 1)
        amplitude = torch.abs(fft_result)

        # 2. 从周期性特征生成路由权重
        period_features = self.period_extractor(amplitude) # (B*C, d_model)
        period_features = F.gelu(period_features)
        
        # 形状为 (B*C, num_wavelet_bands)
        routing_weights = self.router_generator(period_features) 

        # 3. 使用Softmax归一化权重，使其成为概率分布
        # 增加一个维度以用于后续的广播乘法: (B*C, num_wavelet_bands, 1)
        routing_weights = F.softmax(routing_weights, dim=-1).unsqueeze(-1)
        
        return routing_weights

# =====================================================================================
# 3. 主模型 
# =====================================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # --- 基础模型架构参数 ---
        # 建议将这些参数作为configs的一部分传入，这里为了演示方便直接定义
        configs.d_model = getattr(configs, 'd_model', 128)
        configs.n_heads = getattr(configs, 'n_heads', 8)
        configs.d_ff = getattr(configs, 'd_ff', 256)
        configs.dropout = getattr(configs, 'dropout', 0.1)
        configs.num_encoder_layers = getattr(configs, 'num_encoder_layers', 3)

        # --- CoWa-Mixer 特定创新参数 ---
        configs.num_scales = getattr(configs, 'num_scales', 3)
        configs.wavelet = getattr(configs, 'wavelet', 'db4')
        configs.wavelet_level = getattr(configs, 'wavelet_level', 3)
        configs.tokens_mlp_dim = getattr(configs, 'tokens_mlp_dim', 64)

        # --- 其他 ---
        configs.revin = getattr(configs, 'revin', True)
        self.configs = configs
        self.revin_layer = RevIN(configs.enc_in, affine=True) if configs.revin else None

        # --- 小波变换参数 ---
        self.wavelet = configs.wavelet
        self.wavelet_level = configs.wavelet_level
        self.num_wavelet_bands = self.wavelet_level + 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 优化点: 实例化频段路由器
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.frequency_router = FrequencyRouter(
            seq_len=configs.seq_len,
            num_wavelet_bands=self.num_wavelet_bands,
            d_model=configs.d_model
        )

        # --- 金字塔构建与嵌入 ---
        self.temporal_embedders = nn.ModuleList()
        for i in range(configs.num_scales):
            # 动态计算输入长度，避免硬编码
            input_len = (configs.seq_len + 2**i - 1) // (2**i) if i > 0 else configs.seq_len
            self.temporal_embedders.append(nn.Linear(input_len, configs.d_model))

        # 小波频带嵌入层 (每个频带一个)
        self.wavelet_embedders = nn.ModuleList(
            [nn.Linear(configs.seq_len, configs.d_model) for _ in range(self.num_wavelet_bands)]
        )

        # --- 相干门控融合主干 ---
        self.gating_backbone = nn.ModuleList([
            CoherentGatingBlock(configs.d_model, configs.n_heads, configs.dropout)
            for _ in range(configs.num_encoder_layers)
        ])

        # --- 跨尺度混合器 ---
        self.inter_scale_mixer = MixerBlock(configs.num_scales, configs.d_model, configs.tokens_mlp_dim, configs.dropout)

        # --- 预测头 ---
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(), nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L_in, C = x_enc.shape
        if self.revin_layer: x_enc = self.revin_layer(x_enc, 'norm')
        # permute and reshape for channel independence
        x_ci = x_enc.permute(0, 2, 1).reshape(B * C, L_in)

        # --- 1. 构建时域金字塔 (Temporal Pyramid) ---
        temporal_pyramid = []
        for i in range(self.configs.num_scales):
            if i == 0:
                t_s = x_ci
            else:
                # 使用padding确保即使在奇数长度下也能正常工作
                t_s = F.avg_pool1d(x_ci.unsqueeze(1), kernel_size=2**i, stride=2**i, padding=0).squeeze(1)
            temporal_pyramid.append(self.temporal_embedders[i](t_s))
        
        # --- 2. 构建小波金字塔 (Wavelet Pyramid) ---
        with torch.no_grad(): # 小波变换本身不可导，在no_grad下执行
            coeffs = pywt.wavedec(x_ci.cpu().numpy(), self.wavelet, level=self.wavelet_level)
        
        wavelet_bands_embedded = []
        for i, band_coeffs_np in enumerate(coeffs):
            band_coeffs = torch.from_numpy(band_coeffs_np).float().to(x_enc.device)
            aligned_coeffs = F.interpolate(
                band_coeffs.unsqueeze(1), size=L_in, mode='linear', align_corners=False
            ).squeeze(1)
            embedded_band = self.wavelet_embedders[i](aligned_coeffs)
            wavelet_bands_embedded.append(embedded_band)
        
        # h_w_pyramid: (B*C, num_wavelet_bands, D)
        h_w_pyramid = torch.stack(wavelet_bands_embedded, dim=1)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 优化点: 使用频段路由器生成权重，并对小波金字塔进行加权
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 1. 获取路由权重，形状为 (B*C, num_wavelet_bands, 1)
        routing_weights = self.frequency_router(x_ci)
        
        # 2. 对小波金字塔进行加权
        # (B*C, num_wavelet_bands, D) * (B*C, num_wavelet_bands, 1) -> (B*C, num_wavelet_bands, D)
        # 广播机制会自动处理最后一个维度
        h_w_pyramid_weighted = h_w_pyramid * routing_weights

        # --- 3. 相干门控融合 (Coherent Gating Fusion) ---
        h_t_pyramid = [t.unsqueeze(1) for t in temporal_pyramid] # List of (B*C, 1, D)
        
        # 时域金字塔的每个尺度都会查询加权后的、更具动态性的频域金字塔
        for block in self.gating_backbone:
            h_t_pyramid = [block(h_t, h_w_pyramid_weighted) for h_t in h_t_pyramid]

        # --- 4. 跨尺度混合 (Cross-Scale Mixing) ---
        h_t_final = [h.squeeze(1) for h in h_t_pyramid]
        stacked_for_mixer = torch.stack(h_t_final, dim=1) # (B*C, S, D)
        mixed_features = self.inter_scale_mixer(stacked_for_mixer)

        # --- 5. 聚合与预测 (Aggregation & Prediction) ---
        final_repr = mixed_features.mean(dim=1) # (B*C, D)
        prediction = self.prediction_head(final_repr)
        
        # --- 6. 恢复形状与反归一化 ---
        prediction = prediction.reshape(B, C, self.configs.pred_len).permute(0, 2, 1)
        if self.revin_layer:
            prediction = self.revin_layer(prediction, 'denorm')
        
        return prediction