import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


import torch
import torch.nn as nn

class DE(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm_layer = nn.BatchNorm1d(1)  

    def forward(self, y, x, n_C, n_T, training=True):
        if x.shape[-1] == 1:
            y_diff, x_diff, d, x_n, y_n = self.derivative_function(y, x, n_C, n_T)
        else:
            y_diff, x_diff, d, x_n, y_n = self.derivative_function_2d(y, x, n_C, n_T)

        # Replace NaN values with 10000.0
        d_1 = torch.where(torch.isnan(d), torch.tensor(10000.0, device=d.device), d)

        # Replace values with abs > 200 with 0.0
        d_2 = torch.where(torch.abs(d) > 200., torch.tensor(0.0, device=d.device), d)

        # Batch normalization - need to handle shape for BatchNorm1d
        d_reshaped = d_2.reshape(-1, 1)  # Reshape to (batch*seq, 1) for BatchNorm1d
        d_normalized = self.batch_norm_layer(d_reshaped)
        d_normalized = d_normalized.reshape(d_2.shape)  # Reshape back to original

        # Create label tensor
        d_label = (d_2 == d_1).float()

        # Concatenate along the last dimension
        d = torch.cat([d_normalized, d_label], dim=-1)

        return y_diff, x_diff, d, x_n, y_n

    def derivative_function(self, y_values, x_values, context_n, target_m):

        epsilon = 0.000002

        batch_size = y_values.shape[0]
        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]

        # Context section
        current_x = torch.unsqueeze(x_values[:, :context_n], dim=2)
        current_y = torch.unsqueeze(y_values[:, :context_n], dim=2)

        x_temp = x_values[:, :context_n]
        x_temp = torch.repeat_interleave(torch.unsqueeze(x_temp, dim=1), repeats=context_n, dim=1)

        y_temp = y_values[:, :context_n]
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_temp, dim=1), repeats=context_n, dim=1)

        # Calculate Euclidean distances and get indices
        distances = torch.norm((current_x - x_temp), p=2, dim=-1)
        ix = torch.argsort(distances, dim=-1)[:, :, 1]

        # Create selection indices
        batch_indices = torch.repeat_interleave(torch.arange(batch_size * context_n), 1).reshape(-1, 1)
        selection_indices = torch.cat([
            batch_indices,
            ix.reshape(-1, 1)
        ], dim=1)

        # Gather closest points
        x_temp_flat = x_temp.reshape(-1, context_n, dim_x)
        y_temp_flat = y_temp.reshape(-1, context_n, dim_y)

        x_closest = torch.gather(
            x_temp_flat,
            1,
            selection_indices[:, 1].reshape(batch_size, context_n, 1).expand(-1, -1, dim_x)
        )

        y_closest = torch.gather(
            y_temp_flat,
            1,
            selection_indices[:, 1].reshape(batch_size, context_n, 1).expand(-1, -1, dim_y)
        )

        x_rep = current_x[:, :, 0] - x_closest
        y_rep = current_y[:, :, 0] - y_closest

        deriv = y_rep / (epsilon + torch.norm(x_rep, p=2, dim=-1, keepdim=True))

        dydx_dummy = deriv
        diff_y_dummy = y_rep
        diff_x_dummy = x_rep
        closest_y_dummy = y_closest
        closest_x_dummy = x_closest

        # Target selection
        current_x = torch.unsqueeze(x_values[:, context_n:context_n+target_m], dim=2)
        current_y = torch.unsqueeze(y_values[:, context_n:context_n+target_m], dim=2)

        x_temp = torch.repeat_interleave(
            torch.unsqueeze(x_values[:, :target_m+context_n], dim=1),
            repeats=target_m,
            dim=1
        )
        y_temp = torch.repeat_interleave(
            torch.unsqueeze(y_values[:, :target_m+context_n], dim=1),
            repeats=target_m,
            dim=1
        )

        # Create mask
        x_mask = torch.tril(torch.ones((target_m, context_n + target_m), dtype=torch.bool), diagonal=context_n)
        x_mask_inv = ~x_mask
        x_mask_float = x_mask_inv.float() * 1000
        x_mask_float_repeat = torch.repeat_interleave(
            torch.unsqueeze(x_mask_float, dim=0),
            repeats=batch_size,
            dim=0
        )

        # Calculate distances with mask
        distances = torch.norm((current_x - x_temp), p=2, dim=-1).float() + x_mask_float_repeat
        ix = torch.argsort(distances, dim=-1)[:, :, 1]

        # Create selection indices for target section
        batch_indices_target = torch.repeat_interleave(torch.arange(batch_size * target_m), 1).reshape(-1, 1)
        selection_indices_target = torch.cat([
            batch_indices_target,
            ix.reshape(-1, 1)
        ], dim=1)

        # Gather closest points for target section
        x_temp_flat_target = x_temp.reshape(-1, target_m+context_n, dim_x)
        y_temp_flat_target = y_temp.reshape(-1, target_m+context_n, dim_y)

        x_closest_target = torch.gather(
            x_temp_flat_target,
            1,
            selection_indices_target[:, 1].reshape(batch_size, target_m, 1).expand(-1, -1, dim_x)
        )

        y_closest_target = torch.gather(
            y_temp_flat_target,
            1,
            selection_indices_target[:, 1].reshape(batch_size, target_m, 1).expand(-1, -1, dim_y)
        )

        x_rep_target = current_x[:, :, 0] - x_closest_target
        y_rep_target = current_y[:, :, 0] - y_closest_target

        deriv_target = y_rep_target / (epsilon + torch.norm(x_rep_target, p=2, dim=-1, keepdim=True))

        # Concatenate context and target results
        dydx_dummy = torch.cat([dydx_dummy, deriv_target], dim=1)
        diff_y_dummy = torch.cat([diff_y_dummy, y_rep_target], dim=1)
        diff_x_dummy = torch.cat([diff_x_dummy, x_rep_target], dim=1)
        closest_y_dummy = torch.cat([closest_y_dummy, y_closest_target], dim=1)
        closest_x_dummy = torch.cat([closest_x_dummy, x_closest_target], dim=1)

        return diff_y_dummy, diff_x_dummy, dydx_dummy, closest_x_dummy, closest_y_dummy

    def derivative_function_2d(self, y_values, x_values, context_n, target_m):

        epsilon = 0.0000

        def dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2):
            # "z" is the second dim of x input
            numerator = y_closest_2 - current_y[:, :, 0] - ((x_closest_2[:, :, :1] - current_x[:, :, 0, :1]) *
                        (y_closest_1 - current_y[:, :, 0])) / (x_closest_1[:, :, :1] - current_x[:, :, 0, :1] + epsilon)
            denom = x_closest_2[:, :, 1:2] - current_x[:, :, 0, 1:2] - (x_closest_1[:, :, 1:2] - current_x[:, :, 0, 1:2]) * \
                    (x_closest_2[:, :, :1] - current_x[:, :, 0, :1]) / (x_closest_1[:, :, :1] - current_x[:, :, 0, :1] + epsilon)
            dydz_pred = numerator / (denom + epsilon)
            return dydz_pred

        def dydx(dydz_val, current_y, y_closest_1, current_x, x_closest_1):
            dydx_val = (y_closest_1 - current_y[:, :, 0] - dydz_val *
                    (x_closest_1[:, :, 1:2] - current_x[:, :, 0, 1:2])) / \
                    (x_closest_1[:, :, :1] - current_x[:, :, 0, :1] + epsilon)
            return dydx_val

        batch_size = y_values.shape[0]
        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]

        # Context section
        current_x = torch.unsqueeze(x_values[:, :context_n], dim=2)
        current_y = torch.unsqueeze(y_values[:, :context_n], dim=2)

        x_temp = x_values[:, :context_n]
        x_temp = torch.repeat_interleave(torch.unsqueeze(x_temp, dim=1), repeats=context_n, dim=1)

        y_temp = y_values[:, :context_n]
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_temp, dim=1), repeats=context_n, dim=1)

        # Calculate distances and get indices for 1st and 2nd closest points
        distances = torch.norm((current_x - x_temp), p=2, dim=-1)
        ix_1 = torch.argsort(distances, dim=-1)[:, :, 1]
        ix_2 = torch.argsort(distances, dim=-1)[:, :, 2]

        # Create selection indices
        batch_indices = torch.repeat_interleave(torch.arange(batch_size * context_n), 1).reshape(-1, 1)

        selection_indices_1 = torch.cat([
            batch_indices,
            ix_1.reshape(-1, 1)
        ], dim=1)

        selection_indices_2 = torch.cat([
            batch_indices,
            ix_2.reshape(-1, 1)
        ], dim=1)

        # Gather closest points with random noise
        x_temp_flat = x_temp.reshape(-1, context_n, dim_x)
        y_temp_flat = y_temp.reshape(-1, context_n, dim_y)

        x_closest_1 = torch.gather(
            x_temp_flat,
            1,
            selection_indices_1[:, 1].reshape(batch_size, context_n, 1).expand(-1, -1, dim_x)
        ) + torch.randn((batch_size, context_n, dim_x), device=x_values.device) * 0.01

        x_closest_2 = torch.gather(
            x_temp_flat,
            1,
            selection_indices_2[:, 1].reshape(batch_size, context_n, 1).expand(-1, -1, dim_x)
        ) + torch.randn((batch_size, context_n, dim_x), device=x_values.device) * 0.01

        y_closest_1 = torch.gather(
            y_temp_flat,
            1,
            selection_indices_1[:, 1].reshape(batch_size, context_n, 1).expand(-1, -1, dim_y)
        )

        y_closest_2 = torch.gather(
            y_temp_flat,
            1,
            selection_indices_2[:, 1].reshape(batch_size, context_n, 1).expand(-1, -1, dim_y)
        )

        x_rep_1 = current_x[:, :, 0] - x_closest_1
        x_rep_2 = current_x[:, :, 0] - x_closest_2

        y_rep_1 = current_y[:, :, 0] - y_closest_1
        y_rep_2 = current_y[:, :, 0] - y_closest_2

        # Calculate derivatives
        dydx_2 = dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2)
        dydx_1 = dydx(dydx_2, current_y, y_closest_1, current_x, x_closest_1)

        deriv_dummy = torch.cat([dydx_1, dydx_2], dim=-1)
        diff_y_dummy = torch.cat([y_rep_1, y_rep_2], dim=-1)
        diff_x_dummy = torch.cat([x_rep_1, x_rep_2], dim=-1)
        closest_y_dummy = torch.cat([y_closest_1, y_closest_2], dim=-1)
        closest_x_dummy = torch.cat([x_closest_1, x_closest_2], dim=-1)

        # Target selection
        current_x_target = torch.unsqueeze(x_values[:, context_n:context_n+target_m], dim=2)
        current_y_target = torch.unsqueeze(y_values[:, context_n:context_n+target_m], dim=2)

        x_temp_target = torch.repeat_interleave(
            torch.unsqueeze(x_values[:, :target_m+context_n], dim=1),
            repeats=target_m,
            dim=1
        )
        y_temp_target = torch.repeat_interleave(
            torch.unsqueeze(y_values[:, :target_m+context_n], dim=1),
            repeats=target_m,
            dim=1
        )

        # Create mask for target section
        x_mask = torch.tril(torch.ones((target_m, context_n + target_m), dtype=torch.bool), diagonal=context_n)
        x_mask_inv = ~x_mask
        x_mask_float = x_mask_inv.float() * 1000
        x_mask_float_repeat = torch.repeat_interleave(
            torch.unsqueeze(x_mask_float, dim=0),
            repeats=batch_size,
            dim=0
        )

        # Calculate distances with mask
        distances_target = torch.norm((current_x_target - x_temp_target), p=2, dim=-1).float() + x_mask_float_repeat
        ix_1_target = torch.argsort(distances_target, dim=-1)[:, :, 1]
        ix_2_target = torch.argsort(distances_target, dim=-1)[:, :, 2]

        # Create selection indices for target section
        batch_indices_target = torch.repeat_interleave(torch.arange(batch_size * target_m), 1).reshape(-1, 1)

        selection_indices_1_target = torch.cat([
            batch_indices_target,
            ix_1_target.reshape(-1, 1)
        ], dim=1)

        selection_indices_2_target = torch.cat([
            batch_indices_target,
            ix_2_target.reshape(-1, 1)
        ], dim=1)

        # Gather closest points for target section with random noise
        x_temp_flat_target = x_temp_target.reshape(-1, target_m+context_n, dim_x)
        y_temp_flat_target = y_temp_target.reshape(-1, target_m+context_n, dim_y)

        x_closest_1_target = torch.gather(
            x_temp_flat_target,
            1,
            selection_indices_1_target[:, 1].reshape(batch_size, target_m, 1).expand(-1, -1, dim_x)
        ) + torch.randn((batch_size, target_m, dim_x), device=x_values.device) * 0.01

        x_closest_2_target = torch.gather(
            x_temp_flat_target,
            1,
            selection_indices_2_target[:, 1].reshape(batch_size, target_m, 1).expand(-1, -1, dim_x)
        ) + torch.randn((batch_size, target_m, dim_x), device=x_values.device) * 0.01

        y_closest_1_target = torch.gather(
            y_temp_flat_target,
            1,
            selection_indices_1_target[:, 1].reshape(batch_size, target_m, 1).expand(-1, -1, dim_y)
        )

        y_closest_2_target = torch.gather(
            y_temp_flat_target,
            1,
            selection_indices_2_target[:, 1].reshape(batch_size, target_m, 1).expand(-1, -1, dim_y)
        )

        x_rep_1_target = current_x_target[:, :, 0] - x_closest_1_target
        x_rep_2_target = current_x_target[:, :, 0] - x_closest_2_target

        y_rep_1_target = current_y_target[:, :, 0] - y_closest_1_target
        y_rep_2_target = current_y_target[:, :, 0] - y_closest_2_target

        # Calculate derivatives for target section
        dydx_2_target = dydz(current_y_target, y_closest_1_target, y_closest_2_target,
                            current_x_target, x_closest_1_target, x_closest_2_target)
        dydx_1_target = dydx(dydx_2_target, current_y_target, y_closest_1_target,
                            current_x_target, x_closest_1_target)

        deriv_dummy_2 = torch.cat([dydx_1_target, dydx_2_target], dim=-1)
        diff_y_dummy_2 = torch.cat([y_rep_1_target, y_rep_2_target], dim=-1)
        diff_x_dummy_2 = torch.cat([x_rep_1_target, x_rep_2_target], dim=-1)
        closest_y_dummy_2 = torch.cat([y_closest_1_target, y_closest_2_target], dim=-1)
        closest_x_dummy_2 = torch.cat([x_closest_1_target, x_closest_2_target], dim=-1)

        # Concatenate all
        deriv_dummy_full = torch.cat([deriv_dummy, deriv_dummy_2], dim=1)
        diff_y_dummy_full = torch.cat([diff_y_dummy, diff_y_dummy_2], dim=1)
        diff_x_dummy_full = torch.cat([diff_x_dummy, diff_x_dummy_2], dim=1)
        closest_y_dummy_full = torch.cat([closest_y_dummy, closest_y_dummy_2], dim=1)
        closest_x_dummy_full = torch.cat([closest_x_dummy, closest_x_dummy_2], dim=1)

        return diff_y_dummy_full, diff_x_dummy_full, deriv_dummy_full, closest_x_dummy_full, closest_y_dummy_full


class FeatureWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T):
        dim_x = x_n.shape[-1]

        value_x = y.clone()  # identity equivalent
        x_prime = torch.cat([x_emb, x_diff, x_n], dim=2)
        query_x = x_prime.clone()
        key_x = x_prime.clone()

        y_prime = torch.cat([y, y_diff, d, y_n], dim=-1)
        batch_s = y_prime.shape[0]
        key_xy_label = torch.zeros((batch_s, n_C + n_T, 1), device=y_prime.device)
        value_xy = torch.cat([y_prime, key_xy_label, x_prime], dim=-1)
        key_xy = value_xy.clone()

        query_xy_label = torch.cat([
            torch.zeros((batch_s, n_C, 1), device=y_prime.device),
            torch.ones((batch_s, n_T, 1), device=y_prime.device)
        ], dim=1)

        y_prime_masked = torch.cat([
            self.mask_target_pt(y, n_C, n_T),
            self.mask_target_pt(y_diff, n_C, n_T),
            self.mask_target_pt(d, n_C, n_T),
            y_n
        ], dim=2)

        query_xy = torch.cat([y_prime_masked, query_xy_label, x_prime], dim=-1)

        return query_x, key_x, value_x, query_xy, key_xy, value_xy

    def mask_target_pt(self, y, n_C, n_T):
        dim = y.shape[-1]
        batch_s = y.shape[0]

        mask_y = torch.cat([
            y[:, :n_C],
            torch.zeros((batch_s, n_T, dim), device=y.device)
        ], dim=1)
        return mask_y

    def permute(self, x, y, n_C, n_T, num_permutation_repeats):
        if num_permutation_repeats < 1:
            return x, y
        else:
            # Shuffle target only
            x_permuted_list = []
            y_permuted_list = []

            for j in range(num_permutation_repeats):
                # For x
                x_context = x[:, :n_C, :]
                x_target = x[:, n_C:, :]
                # Permute target points across batch dimension
                x_target_permuted = x_target[:, torch.randperm(x_target.shape[1]), :]
                x_combined = torch.cat([x_context, x_target_permuted], dim=1)
                x_permuted_list.append(x_combined)

                # For y
                y_context = y[:, :n_C, :]
                y_target = y[:, n_C:, :]
                # Permute target points across batch dimension
                y_target_permuted = y_target[:, torch.randperm(y_target.shape[1]), :]
                y_combined = torch.cat([y_context, y_target_permuted], dim=1)
                y_permuted_list.append(y_combined)

            x_permuted = torch.cat(x_permuted_list, dim=0)
            y_permuted = torch.cat(y_permuted_list, dim=0)

            return x_permuted, y_permuted

    def PE(self, x, enc_dim, xΔmin, xmax):
        R = xmax / xΔmin * 100

        # Create frequency ranges
        even_indices = torch.arange(0, enc_dim, 2, device=x.device)
        odd_indices = torch.arange(1, enc_dim, 2, device=x.device)

        drange_even = (xΔmin * R**(even_indices / enc_dim)).float()
        drange_odd = (xΔmin * R**((odd_indices - 1) / enc_dim)).float()

        # Expand dimensions for broadcasting
        drange_even = drange_even.view(1, 1, -1)
        drange_odd = drange_odd.view(1, 1, -1)

        # Apply positional encoding
        sin_component = torch.sin(x / drange_even)
        cos_component = torch.cos(x / drange_odd)

        # Interleave sin and cos components
        result = torch.zeros(x.shape[0], x.shape[1], enc_dim, device=x.device)
        result[:, :, 0::2] = sin_component
        result[:, :, 1::2] = cos_component

        return result