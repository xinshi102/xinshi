# 使models成为一个Python包
from .modle import WeatherForecastModel, WeightedMSELoss, prepare_data
from ..new_models.hour_modle import WeatherLSTM, WeatherDataProcessor

__all__ = ['WeatherForecastModel', 'WeightedMSELoss', 'prepare_data', 'WeatherLSTM', 'WeatherDataProcessor'] 