import os
import math as m
import numpy as np

import torch

from icefreearcticml.icefreearcticml.utils import extend_and_fill_series
from icefreearcticml.icefreearcticml.taylortransformer.pipeline import TaylorFormerPipeline


def nll(y, μ, log_σ, ϵ=0.001):
    pi = torch.tensor(m.pi, dtype=torch.float32)
    y = y.float()

    # Same calculations as TF version
    mse_per_point = torch.square(y - μ)
    σ = torch.exp(log_σ)
    lik_per_point = (1 / 2) * torch.divide(mse_per_point, torch.square(σ + ϵ)) + \
                    torch.log(σ + ϵ) + \
                    (1/2) * torch.log(2*pi)

    sum_lik = torch.sum(lik_per_point)
    sum_mse = torch.sum(mse_per_point)

    return (lik_per_point, sum_mse, sum_lik,
            torch.mean(lik_per_point), torch.mean(mse_per_point))

def train_step(taylorformer_model, optimizer, x, y, n_C, n_T, training=True):
    optimizer.zero_grad()
    μ, log_σ = taylorformer_model(x, y, n_C, n_T, training)
    _, _, _, likpp, mse = nll(y[:, n_C:n_T+n_C], μ, log_σ)

    likpp.backward()
    optimizer.step()

    return μ, log_σ, likpp, mse

def generate_emulation(model, x_data, y_data, total_years=70, full_window=45, prediction_chunk=10):
    model.eval()

    all_predictions = []
    all_uncertainties = []

    x_context = x_data[:, :full_window]
    y_context = y_data

    years_predicted = 0

    with torch.no_grad():
        while years_predicted < total_years:
            n_T = min(prediction_chunk, total_years - years_predicted)
            n_C = full_window - n_T

            x_context = x_data[:, years_predicted:full_window+years_predicted]
            
            μ, log_σ = model(x_context, y_context, n_C, n_T, training=False)
            
            all_predictions.append(μ.cpu().numpy())
            all_uncertainties.append(torch.exp(log_σ).cpu().numpy())
            
            years_predicted += n_T

            y_context = torch.cat([y_context[:, n_T:], μ], dim=1)

    return np.concatenate(all_predictions, axis=1), np.concatenate(all_uncertainties, axis=1)


# Define horizon categories for monitoring
def get_horizon_category(n_T):
    if n_T <= 15: return "short"
    elif n_T <= 25: return "medium" 
    else: return "long"

def train_step(taylorformer_model, optimizer, x, y, n_C, n_T, training=True):
    μ, log_σ = taylorformer_model(x, y, n_C, n_T, training)
    _, _, _, likpp, mse = nll(y[:, n_C:n_T+n_C], μ, log_σ)

    return μ, log_σ, likpp, mse

def test_step(taylorformer_model, x, y, n_C, n_T):
    with torch.no_grad():
        μ, log_σ = taylorformer_model(x, y, n_C, n_T, training=False)
        _, _, _, likpp, mse = nll(y[:, n_C:n_T+n_C], μ, log_σ)
    return μ, log_σ, likpp, mse

def split_data_random(n, train_ratio=0.8, val_ratio=0.1):
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx

def prepare_data(model_data, variables, model_name, train_ratio, val_ratio, obs_start=1979, obs_end=2023):
    n = len(model_data[variables[0]][model_name].columns)
    train_idx, val_idx, test_idx = split_data_random(n, train_ratio, val_ratio)

    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for var in variables:
        obs_series = extend_and_fill_series(model_data[var]["Observations"], start_year=obs_start, end_year=obs_end)
        model_series = model_data[var][model_name].loc[obs_series.index].fillna(0)

        x_data = model_series.values.T
        n = x_data.shape[0]
        y_data = np.repeat(obs_series.values.reshape(-1, 1), repeats=n, axis=1).T
        
        # Features (model data)
        x_train.append(x_data[train_idx][:, :, np.newaxis])
        x_val.append(x_data[val_idx][:, :, np.newaxis])
        x_test.append(x_data[test_idx][:, :, np.newaxis])
        
        # Targets (observations)
        y_train.append(y_data[train_idx][:, :, np.newaxis])
        y_val.append(y_data[val_idx][:, :, np.newaxis])
        y_test.append(y_data[test_idx][:, :, np.newaxis])

    return {
        "x_train": torch.tensor(np.concatenate(x_train, axis=2), dtype=torch.float32),
        "y_train": torch.tensor(np.concatenate(y_train, axis=2), dtype=torch.float32),
        "x_val": torch.tensor(np.concatenate(x_val, axis=2), dtype=torch.float32),
        "y_val": torch.tensor(np.concatenate(y_val, axis=2), dtype=torch.float32),
        "x_test": torch.tensor(np.concatenate(x_test, axis=2), dtype=torch.float32),
        "y_test": torch.tensor(np.concatenate(y_test, axis=2), dtype=torch.float32),
    }

def setup_model_and_optimiser(
        num_heads=1,
        projection_shape_for_head=8,
        output_shape=1,
        rate=0.1,
        num_layers=4,
        enc_dim=32,
        permutation_repeats=1,
        epochs=10000,
    ):
    model = TaylorFormerPipeline(
        num_heads=num_heads,
        projection_shape_for_head=projection_shape_for_head,
        output_shape=output_shape,
        rate=rate,
        num_layers=num_layers,
        enc_dim=enc_dim,
        permutation_repeats=permutation_repeats,
    )
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=2000)
    warmup_iters = int(epochs * 0.1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=0.1, total_iters=warmup_iters)
    
    return model, optimiser, scheduler, warmup_scheduler

def train_epoch(model, optimiser, x_train, y_train, config):
    n = x_train.shape[0]

    n_C = np.random.randint(config['min_context'], min(n - 10, config['max_context']) + 1)
    n_T = min(n, config['total_length']) - n_C
    horizon_type = get_horizon_category(n_T)

    idx = np.random.choice(n, min(config['batch_size'], n), replace=False)
    x_batch, y_batch = x_train[idx], y_train[idx]

    optimiser.zero_grad()
    μ_train, _, train_lik, train_mse = train_step(model, optimiser, x_batch, y_batch, n_C, n_T, training=True)

    train_mse_scaled = train_mse / config['accumulation_steps']
    train_lik.backward()

    return n_C, n_T, horizon_type, train_mse_scaled

def update_optimiser(model, optimiser, scheduler, warmup_scheduler, epoch, config):
    if (epoch + 1) % config['accumulation_steps'] == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if epoch < (config['epochs'] * config['warmup_ratio']):
            warmup_scheduler.step()
        else:
            scheduler.step()

def validate_model(model, x_val, y_val, n_C, n_T):
    model.eval()
    with torch.no_grad():
        μ_val, log_σ_val = model(x_val, y_val, n_C, n_T, training=False)
        _, _, _, val_likpp, val_mse = nll(y_val[:, n_C:n_C+n_T], μ_val, log_σ_val)
    model.train()
    return val_mse.item()

def save_best_model(model_tag, model, best_val_loss, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': best_val_loss,
    }, f'checkpoints/best_model_{model_tag}.pth')


def train_model(data_res, config, epochs=5000, model_tag="", **model_kwargs):
    model, optimiser, scheduler, warmup_scheduler = setup_model_and_optimiser(**model_kwargs)
    
    train_losses = {'short': [], 'medium': [], 'long': [], 'all': []}
    val_losses = {'short': [], 'medium': [], 'long': [], 'all': []}
    learning_rates = []

    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        
        n_C, n_T, horizon_type, train_mse_scaled = train_epoch(
            model, optimiser, data_res["x_train"], data_res["y_train"], config
        )
        update_optimiser(model, optimiser, scheduler, warmup_scheduler, epoch, config)
        
        train_loss = train_mse_scaled.item() * config['accumulation_steps']
        train_losses[horizon_type].append(train_loss)
        train_losses['all'].append(train_loss)

        val_mse = validate_model(model, data_res["x_val"], data_res["y_val"], n_C, n_T)
        val_losses[horizon_type].append(val_mse)
        val_losses['all'].append(val_mse)

        if val_mse < best_val_loss:
            save_best_model(model_tag, model, best_val_loss, epoch)
        
        learning_rates.append(optimiser.param_groups[0]['lr'])

        if epoch % 100 == 0:
            lr = optimiser.param_groups[0]['lr']
            print(f"Epoch {epoch}: Train MSE = {train_loss:.4f}, "
                  f"Val MSE = {val_mse:.4f}, LR = {lr:.2e}, "
                  f"Horizon = {horizon_type}(n_T={n_T})")

    return {
        'model': model.state_dict(),
        'model_kwargs': model_kwargs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'config': config,
    }

