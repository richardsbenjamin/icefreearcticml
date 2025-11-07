import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys, values, d_k, mask=None):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            inverse_mask = ~mask
            scores = scores.masked_fill(inverse_mask, -1e9)

        weights = F.softmax(scores, dim=-1)

        if mask is not None:
            weights = torch.min(torch.abs(mask.float()), torch.abs(weights))

        return torch.matmul(weights, values)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, output_shape, projection_shape):
        super().__init__()
        self.attention = DotProductAttention()
        self.heads = num_heads
        self.projection_shape = projection_shape
        self.W_q = nn.LazyLinear(projection_shape)
        self.W_k = nn.LazyLinear(projection_shape)
        self.W_v = nn.LazyLinear(projection_shape)
        self.W_o = nn.LazyLinear(output_shape)

        assert projection_shape % self.heads == 0, "heads must be a factor of projection_shape"

    def reshape_tensor(self, x, heads, flag):
        if flag:
            batch_size, seq_length, _ = x.shape
            x = x.view(batch_size, seq_length, heads, -1)
            x = x.transpose(1, 2)  # (batch_size, heads, seq_length, -1)
        else:
            x = x.transpose(1, 2)  # (batch_size, seq_length, heads, -1)
            batch_size, seq_length, heads, features = x.shape
            x = x.contiguous().view(batch_size, seq_length, -1)
        return x

    def forward(self, queries, keys, values, mask=None):
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)

        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.projection_shape // self.heads, mask)
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        return self.W_o(output)


class FFN_1(nn.Module):
    def __init__(self, output_shape, dropout_rate=0.1):
        super(FFN_1, self).__init__()

        self.dense_a = nn.LazyLinear(output_shape)
        self.dense_b = nn.LazyLinear(output_shape)
        self.dense_c = nn.LazyLinear(output_shape)
        self.layernorm = nn.ModuleList([nn.LayerNorm(output_shape) for _ in range(2)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, query, training=True):
        query = self.dense_a(query)
        x = x + query
        x = self.layernorm[0](x)
        x_skip = x.clone()
        x = self.dense_b(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.dense_c(x)
        x = x + x_skip
        return self.layernorm[1](x)


class FFN_o(nn.Module):
    def __init__(self, output_shape, dropout_rate=0.1):
        super(FFN_o, self).__init__()

        self.dense_b = nn.LazyLinear(output_shape)
        self.dense_c = nn.LazyLinear(output_shape)
        self.layernorm = nn.ModuleList([nn.LayerNorm(output_shape) for _ in range(2)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, query, training=True):
        x = x + query
        x = self.layernorm[0](x)
        x_skip = x.clone()
        x = self.dense_b(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.dense_c(x)
        x = x + x_skip
        return self.layernorm[1](x)


class MHA_X_a(nn.Module):
    def __init__(self, num_heads, projection_shape, output_shape, dropout_rate=0.1):
        super(MHA_X_a, self).__init__()
        self.mha = MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_1(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)
        return x


class MHA_XY_a(nn.Module):
    def __init__(self, num_heads, projection_shape, output_shape, dropout_rate=0.1):
        super(MHA_XY_a, self).__init__()
        self.mha = MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_1(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)
        return x


class MHA_X_b(nn.Module):
    def __init__(self, num_heads, projection_shape, output_shape, dropout_rate=0.1):
        super(MHA_X_b, self).__init__()
        self.mha = MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_o(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)
        return x


class MHA_XY_b(nn.Module):
    def __init__(self, num_heads, projection_shape, output_shape, dropout_rate=0.1):
        super(MHA_XY_b, self).__init__()
        self.mha = MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_o(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)
        return x


class TaylorFormer(nn.Module):
    def __init__(self, num_heads, projection_shape, output_shape, num_layers,
                 dropout_rate=0.1, target_y_dim=1, bound_std=False):
        super().__init__()

        self.num_layers = num_layers

        self.mha_x_a = MHA_X_a(num_heads, projection_shape, output_shape, dropout_rate=dropout_rate)

        self.mha_x_b = nn.ModuleList([
            MHA_X_b(num_heads, projection_shape, output_shape, dropout_rate=dropout_rate)
            for _ in range(num_layers-1)
        ])

        self.mha_xy_a = MHA_XY_a(num_heads, projection_shape, output_shape, dropout_rate=dropout_rate)

        self.mha_xy_b = nn.ModuleList([
            MHA_XY_b(num_heads, projection_shape, output_shape, dropout_rate=dropout_rate)
            for _ in range(num_layers-1)
        ])

        self.linear_layer = nn.Linear(2 * output_shape, output_shape)
        self.dense_sigma = nn.Linear(output_shape, target_y_dim)
        self.dense_last = nn.Linear(output_shape, target_y_dim)
        self.bound_std = bound_std

    def forward(self, query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n, training=True):
        x = self.mha_x_a(query_x, query_x, query_x, mask, training=training)
        xy = self.mha_xy_a(query_xy, key_xy, value_xy, mask, training=training)

        for i in range(self.num_layers - 2):
            xy = self.mha_xy_b[i](xy, xy, xy, mask, training=training)
            x = self.mha_x_b[i](x, x, x, mask, training=training)

        if self.num_layers > 1:
            xy = self.mha_xy_b[-1](xy, xy, xy, mask, training=training)
            x = self.mha_x_b[-1](x, x, value_x, mask, training=training)

        combo = torch.cat([x, xy], dim=2)
        z = self.linear_layer(combo)

        log_σ = self.dense_sigma(z)
        μ = self.dense_last(z) + y_n

        if self.bound_std:
            σ = 0.01 + 0.99 * F.softplus(log_σ)
        else:
            σ = torch.exp(log_σ)

        log_σ = torch.log(σ)

        return μ, log_σ