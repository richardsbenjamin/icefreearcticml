import numpy as np
from pandas import DataFrame, to_datetime
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from lightning.pytorch import Trainer


def get_timeseries_dataloader(train_config, data, train = True):
    dataset = TimeSeriesDataSet(
        data,
        time_idx="time", 
        group_ids=["member"],
        target=train_config.y_var,
        min_encoder_length=train_config.max_encoder_length,
        max_encoder_length=train_config.max_encoder_length,
        min_prediction_length=train_config.max_prediction_length,
        max_prediction_length=train_config.max_prediction_length,
        time_varying_known_reals=["time"],
        time_varying_unknown_reals=train_config.x_vars,
        static_categoricals=["member"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset.to_dataloader(train=train, batch_size=32, num_workers=0)

def train_model(train_dataloader, max_epochs=15):
    model = TemporalFusionTransformer.from_dataset(train_dataloader.dataset)
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", enable_progress_bar=False)
    trainer.fit(model, train_dataloader)
    return model

def predictions_to_dataframe(index, output_df):
    prediction_data = []
    
    for i, idx_time, idx_member, in index.to_records():
        pred_sequence = output_df[i]
        
        start_year = idx_time + 1
        years = range(start_year, start_year + len(pred_sequence))
        
        for year_idx, year in enumerate(years):
            prediction_data.append({
                'member': idx_member,
                'time': year,
                'prediction': pred_sequence[year_idx].item(),
                'encoder_end_year': idx_time
            })
    
    return DataFrame(prediction_data)

def run_model_from_config(train_config, model_data):
    train_config.set_all_data(model_data)
    train_loader = get_timeseries_dataloader(train_config, train_config.train_data, train=True)
    test_loader = get_timeseries_dataloader(train_config, train_config.test_data, train=False)
    model = train_model(train_loader, train_config.epochs)
    predictions = model.predict(test_loader, return_index=True)

    if isinstance(predictions.output, list):
        results = {}
        for var, output in zip(train_config.y_var, predictions.output):
            results[var] = predictions_to_dataframe(predictions.index, output)
    else:
        results = predictions_to_dataframe(predictions.index, predictions.output)

    return {
        "model_dict": model.state_dict(),
        "train_config": train_config,
        "results": results,
    }

def unmelt(melted: DataFrame) -> DataFrame:
    df_pivot = melted.pivot(index='time', columns='member', values='prediction')
    df_pivot.index = to_datetime(df_pivot.index, format='%Y')
    return df_pivot

def detrend_with_trend(df: DataFrame, train_members: list) -> tuple:
    residuals_df = DataFrame(index=df.index, columns=df.columns)
    trend_df = DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        series = df[col]
        # Fit linear trend using polyfit
        coeffs = np.polyfit(range(len(series)), series, 1)
        trend = np.polyval(coeffs, range(len(series)))
        
        # Calculate residuals and store
        residuals_df[col] = series - trend
        trend_df[col] = trend
    
    return residuals_df, trend_df

def calculate_metrics(pred, truth):
    return {
        'rmse': np.sqrt(((pred - truth) ** 2).mean()),
        'mae': (pred - truth).abs().mean(),
        'mse': ((pred - truth) ** 2).mean(),
        'correlation': pred.corr(truth),
    }

class PredictionComparator:
    def __init__(self, predictions_df, ground_truth_df):
        self.predictions = predictions_df
        self.ground_truth = ground_truth_df
        self.ground_truth_long = self._prepare_ground_truth()
    
    def _prepare_ground_truth(self):
        return (
                self.ground_truth.reset_index().melt(
                id_vars=['time'], 
                value_name='ground_truth', 
                var_name='member'
            ).assign(member=lambda x: x['member'].astype(int))
            .assign(time=lambda x: x["time"].dt.year)
        )
    
    def compare_single_series(self, member, encoder_end_year=None):
        if encoder_end_year is not None:
            pred_subset = self.predictions[
                (self.predictions['member'] == member) & 
                (self.predictions['encoder_end_year'] == encoder_end_year)
            ]
        else:
            pred_subset = self.predictions[self.predictions['member'] == member]
        
        comparison = pred_subset.merge(
            self.ground_truth_long,
            on=['member', 'time'],
            how='inner'
        )
        
        metrics = calculate_metrics(comparison['prediction'], comparison['ground_truth'])
        return comparison, metrics
    
    def compare_merged_series(self, member, method='mean'):
        # Get all predictions for this member
        member_predictions = self.predictions[self.predictions['member'] == member]
        
        # Merge overlapping predictions
        if method == 'mean':
            merged_pred = member_predictions.groupby('time')['prediction'].mean().reset_index()
        elif method == 'median':
            merged_pred = member_predictions.groupby('time')['prediction'].median().reset_index()
        else:
            merged_pred = member_predictions.groupby('time')['prediction'].first().reset_index()
        
        # Add member column back
        merged_pred['member'] = member
        
        # Merge with ground truth
        comparison = merged_pred.merge(
            self.ground_truth_long,
            on=['member', 'time'],
            how='inner'
        )
        
        metrics = calculate_metrics(comparison['prediction'], comparison['ground_truth'])
        return comparison, metrics
    
    def compare_all_members(self, method='individual'):
        """Compare all members"""
        results = {}
        
        if method == 'individual':
            # Compare each prediction series individually
            for (member, encoder_year), group in self.predictions.groupby(['member', 'encoder_end_year']):
                comparison, metrics = self.compare_single_series(member, encoder_year)
                results[(member, encoder_year)] = {'comparison': comparison, 'metrics': metrics}
        
        elif method == 'merged':
            # Compare merged predictions for each member
            for member in self.predictions['member'].unique():
                comparison, metrics = self.compare_merged_series(member)
                results[member] = {'comparison': comparison, 'metrics': metrics}
        
        return results
    
