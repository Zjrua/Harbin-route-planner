"""中文旅游游记爬虫（扩充真实哈尔滨路线数据）.

目标源：
- mafengwo_crawler: 马蜂窝游记（行程段落）
- ctrip_crawler: 携程路书（可选）

⚠️ 重要说明：
这些爬虫受以下限制：
1. 目标网站有严格反爬（验证码、IP 封禁、登录态要求）
2. 当前开发环境网络受限（SSL 吊销检查失败、部分域名不可达）
3. 实际使用前需在目标网站可访问的环境验证，并遵守 robots.txt

输出格式与 data/raw/哈尔滨旅游路线数据.csv 一致：
source, title, season, route, liked_count, note_id

其中 route 字段为 "地名→地名→地名" 文本序列，
后续复用 prepare_data.py 的 load_xhs_routes 做 POI 名称匹配转索引。
"""

from .mafengwo_crawler import MafengwoCrawler

__all__ = ["MafengwoCrawler"]
