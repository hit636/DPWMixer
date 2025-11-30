import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================================
# 1. 基础组件 (RevIN)
# =====================================================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    用于解决分布偏移，保证模型能专注于形态而非绝对数值。
    """
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
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev + self.mean
        return x

# =====================================================================================
# 2. 频域/多尺度处理: Haar Wavelet
# =====================================================================================

class HaarWaveletSplit(nn.Module):
    """
    Haar 小波分解。
    相比 Pooling，它能将信息无损分离为 Low(Trend) 和 High(Detail)。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        inv_sqrt2 = 1.0 / np.sqrt(2)
        # 注册不可学习的卷积核
        self.register_buffer('low_filter', torch.tensor([[[inv_sqrt2, inv_sqrt2]]]))
        self.register_buffer('high_filter', torch.tensor([[[inv_sqrt2, -inv_sqrt2]]]))

    def forward(self, x):
        # x: [Batch, Length, Channel]
        B, L, C = x.shape
        
        # 填充以确保偶数长度
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        
        x = x.permute(0, 2, 1) # [B, C, L]
        
        # Group Conv 实现独立通道分解
        low = F.conv1d(x, self.low_filter.repeat(C, 1, 1), stride=2, groups=C)
        high = F.conv1d(x, self.high_filter.repeat(C, 1, 1), stride=2, groups=C)
        
        return low.permute(0, 2, 1), high.permute(0, 2, 1)

# =====================================================================================
# 3. 核心替换模块: Dual-Path Trend Mixer
# =====================================================================================

class DualTrendMixer(nn.Module):
    """
    1. Global Trend Path: 纯线性映射 (DLinear思想)，捕捉整体单调性/线性趋势。
    2. Local Evolution Path (Patch+MLP): 捕捉平滑的非线性趋势变化。
    """
    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8, d_model=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # --- Path 1: Global Linear Trend (刚性趋势) ---
        # 这种简单的层在 LTSF 任务中对 Trend 的捕捉通常是 SOTA 的
        self.global_linear = nn.Linear(seq_len, pred_len)
        
        # --- Path 2: Local Evolution (柔性趋势) ---
        # 使用 Patching 减少计算量并提取局部语义
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # MLP Mixer (Bottleneck结构)
        # 维度变换: D_model -> D_ff -> D_model
        d_ff = d_model * 2
        self.mlp_mixer = nn.Sequential(
            nn.Linear(d_model * self.num_patches, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len) # 直接映射到预测长度
        )
        self.dropout = nn.Dropout(dropout)

        # 融合门控 (可选，这里直接相加效果通常最好)
        self.w_global = nn.Parameter(torch.tensor(0.6)) # 初始化偏向 Global
        self.w_local = nn.Parameter(torch.tensor(0.4))

    def forward(self, x):
        # x: [Batch, Length, Channel]
        B, L, C = x.shape
        
        # Channel Independence: 处理 reshape 为 [B*C, L]
        x_reshaped = x.permute(0, 2, 1).reshape(B * C, L)
        
        # === 1. Global Trend Path ===
        trend_out = self.global_linear(x_reshaped) # [B*C, Pred_Len]
        
        # === 2. Local Evolution Path ===
        # Patching: [B*C, L] -> [B*C, 1, L] -> Unfold
        x_in = x_reshaped.unsqueeze(1)
        patches = x_in.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.squeeze(1) # [B*C, Num_Patches, Patch_Len]
        
        # Embed
        p_out = self.patch_embedding(patches) # [B*C, Num, D]
        p_out = p_out + self.pos_embedding
        p_out = self.dropout(p_out)
        
        # Flatten & Mix
        p_flat = p_out.reshape(B * C, -1)
        local_out = self.mlp_mixer(p_flat) # [B*C, Pred_Len]
        
        # === Fusion ===
        # 动态加权求和
        out = self.w_global * trend_out + self.w_local * local_out
        
        # Reshape back to [B, Pred_Len, Channel]
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.down_sampling_layers = getattr(configs, 'down_sampling_layers', 3)
        
        # Patching 超参
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.d_model = getattr(configs, 'd_model', 64)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 1. RevIN
        self.revin = RevIN(self.enc_in)
        
        # 2. Wavelet Downsampling (频域分解)
        self.wave_modules = nn.ModuleList()
        for i in range(self.down_sampling_layers):
            self.wave_modules.append(HaarWaveletSplit(self.enc_in))
            
        # 3. Multi-Scale Trend Mixers
        self.mixers = nn.ModuleList()
        
        # Scale 0 (原始分辨率)
        self.mixers.append(
            DualTrendMixer(self.seq_len, self.pred_len, self.enc_in, 
                           self.patch_len, self.stride, self.d_model, self.dropout)
        )
        
        # Scale 1..N (下采样)
        curr_len = self.seq_len
        for i in range(self.down_sampling_layers):
            curr_len = int(np.ceil(curr_len / 2))
            # 注意: Patch Size 和 Stride 对于很短的序列可能需要调整，这里为简化保持不变
            # 实际工程中可能需要 if curr_len < patch_len: padding
            
            # 对于下采样层，我们使用较小的 d_model 节省计算
            self.mixers.append(
                DualTrendMixer(curr_len, self.pred_len, self.enc_in,
                               self.patch_len, self.stride, self.d_model, self.dropout)
            )
            
        # 4. 自适应融合 (Adaptive Fusion)
        self.fusion_weights = nn.Parameter(torch.ones(self.down_sampling_layers + 1, self.enc_in))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [Batch, Seq_Len, Vars]
        
        # 1. Normalize
        x_enc = self.revin(x_enc, 'norm')
        
        # 2. Wavelet Pyramid
        inputs_scale = [x_enc]
        current_x = x_enc
        
        for wave_op in self.wave_modules:
            low, high = wave_op(current_x)
            current_x = low
            inputs_scale.append(current_x)
            
        # 3. Parallel Trend Mixing
        outs = []
        for i, mixer in enumerate(self.mixers):
            inp = inputs_scale[i]
            # 特殊处理：如果下采样后长度太短，小于 PatchLen，进行 Padding
            if inp.shape[1] < self.patch_len:
                pad_len = self.patch_len - inp.shape[1]
                inp = F.pad(inp, (0, 0, 0, pad_len)) # pad length dim
                
            out = mixer(inp)
            outs.append(out)
            
        # 4. Adaptive Fusion
        # Stack: [Scales, B, Pred_Len, C]
        outs_stack = torch.stack(outs, dim=0)
        
        # Weights: [Scales, C] -> [Scales, 1, 1, C]
        w = self.softmax(self.fusion_weights).unsqueeze(1).unsqueeze(1)
        
        # Sum
        final_pred = torch.sum(outs_stack * w, dim=0)
        
        # 5. Denormalize
        final_pred = self.revin(final_pred, 'denorm')
        
        return final_pred