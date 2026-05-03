"""游客评论爬取模块.

从旅游平台（携程/马蜂窝/大众点评）爬取游客评论，
用于计算 POI 满意度评分和情感分析。
"""

import pandas as pd
from typing import Optional


def crawl_reviews(poi_name: str, platform: str = "ctrip",
                  max_reviews: int = 100) -> pd.DataFrame:
    """爬取指定 POI 的游客评论.

    Args:
        poi_name: POI 名称
        platform: 评论平台 ("ctrip" / "mafengwo" / "dianping")
        max_reviews: 最大评论数

    Returns:
        评论 DataFrame，包含 text, rating, date 等列
    """
    raise NotImplementedError("需实现：爬取评论数据")


def sentiment_score(reviews: pd.DataFrame) -> float:
    """计算评论的情感得分.

    Args:
        reviews: 评论 DataFrame

    Returns:
        0-1 之间的情感得分
    """
    raise NotImplementedError("需实现：基于评论内容计算情感得分")
