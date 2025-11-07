import torch
import torch.nn as nn

from icefreearcticml.icefreearcticml.taylorformer import TaylorFormer
from icefreearcticml.icefreearcticml.taylorformer import FeatureWrapper, DE


class TaylorFormerPipeline(nn.Module):

    def __init__(self, num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                 bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=2, MHAX="xxx", **kwargs):
        super().__init__(**kwargs)

        self._permutation_repeats = permutation_repeats
        self.enc_dim = enc_dim
        self.xmin = xmin
        self.xmax = xmax
        self._feature_wrapper = FeatureWrapper()

        if MHAX == "xxx":
            self._taylorformer = TaylorFormer(
                num_heads=num_heads,
                dropout_rate=rate,
                num_layers=num_layers,
                output_shape=output_shape,
                projection_shape=projection_shape_for_head * num_heads,
                bound_std=bound_std
            )
        self._DE = DE()

    def forward(self, x, y, n_C, n_T, training=True):
        x = x[:, :n_C + n_T, :]
        y = y[:, :n_C + n_T, :]

        if training:
            x, y = self._feature_wrapper.permute(x, y, n_C, n_T, self._permutation_repeats)

        x_emb_list = []
        for i in range(x.shape[-1]):
            x_feature = x[:, :, i].unsqueeze(-1)
            x_emb_feature = self._feature_wrapper.PE(x_feature, self.enc_dim, self.xmin, self.xmax)
            x_emb_list.append(x_emb_feature)
        x_emb = torch.cat(x_emb_list, dim=-1)

        batch_size = x.shape[0]
        seq_len = n_C + n_T

        # Create context part mask
        context_part = torch.cat([
            torch.ones((n_C, n_C), dtype=torch.bool, device=x.device),
            torch.zeros((n_C, n_T), dtype=torch.bool, device=x.device)
        ], dim=-1)

        # Create lower triangular mask
        diagonal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        lower_diagonal_mask = diagonal_mask.clone()
        # Remove diagonal
        lower_diagonal_mask = lower_diagonal_mask & ~torch.eye(seq_len, dtype=torch.bool, device=x.device)

        # Combine masks
        target_part = lower_diagonal_mask[n_C:seq_len, :seq_len]
        mask = torch.cat([context_part, target_part], dim=0)

        # Expand mask for batch dimension if needed by the TaylorFormer
        mask = mask.unsqueeze(0)  # Add batch dimension

        ######## create derivative ########
        y_diff, x_diff, d, x_n, y_n = self._DE(y, x, n_C, n_T, training)

        inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T]
        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(*inputs_for_processing)

        y_n_closest = y_n[:, :, :y.shape[-1]]

        μ, log_σ = self._taylorformer(
            query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest, training=training
        )

        return μ[:, n_C:], log_σ[:, n_C:]