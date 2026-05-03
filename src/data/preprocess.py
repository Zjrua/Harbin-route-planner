"""数据清洗与特征工程."""

import pandas as pd
import numpy as np
from typing import Optional


def clean_poi_data(raw_pois: pd.DataFrame) -> pd.DataFrame:
    """清洗 POI 原始数据.

    处理缺失值、重复项、异常坐标等。

    Args:
        raw_pois: 原始 POI DataFrame

    Returns:
        清洗后的 POI DataFrame
    """
    raise NotImplementedError("需实现：去重、缺失值处理、坐标异常过滤")


def extract_poi_features(pois: pd.DataFrame) -> np.ndarray:
    """提取 POI 特征向量.

    特征包括：评分、类别 one-hot、季节性权重、 popularity 等。

    Args:
        pois: 清洗后的 POI DataFrame

    Returns:
        特征矩阵 [n_pois, feature_dim]
    """
    raise NotImplementedError("需实现：拼接多维度特征")


def normalize_features(features: np.ndarray) -> np.ndarray:
    """标准化特征.

    Args:
        features: 特征矩阵

    Returns:
        标准化后的特征矩阵
    """
    raise NotImplementedError("需实现：StandardScaler 或 MinMaxScaler")
