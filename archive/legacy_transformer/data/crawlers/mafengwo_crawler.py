"""马蜂窝游记爬虫（哈尔滨路线）.

⚠️ 反爬限制说明（必读）：
马蜂窝对爬虫有严格防护，包括：
1. 动态验证码（滑动/点选）
2. IP 频率限制（短时间大量请求会被封）
3. 登录态要求（部分内容需登录可见）
4. User-Agent / Referer 检测

实际运行前必须：
- 在能正常访问 mafengwo.cn 的环境执行
- 配置合理的请求间隔（≥3秒）和 UA 轮换
- 遵守 robots.txt，仅用于个人学术研究
- 小批量试爬验证可行性，再决定是否扩大规模

本爬虫提供完整骨架，但不保证在所有环境下可用。
"""

import csv
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# 哈尔滨-相关游记搜索关键词
HARBIN_KEYWORDS = [
    "哈尔滨旅游攻略",
    "哈尔滨自由行",
    "哈尔滨冰雪大世界",
    "哈尔滨路线",
    "哈尔滨一日游",
    "哈尔滨三日游",
]

DEFAULT_HEADERS = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.mafengwo.cn/",
    },
    # 可加更多 UA 做轮换
]


class MafengwoCrawler:
    """马蜂窝游记爬虫.

    工作流程：
    1. search(): 按关键词搜索游记列表，获取游记 ID 和 URL
    2. fetch_notes(): 逐个抓取游记正文
    3. extract_route(): 从正文中提取"地名→地名"的行程段落
    4. save(): 输出为 CSV（与 XHS 数据格式一致）
    """

    BASE = "https://www.mafengwo.cn"
    SEARCH_URL = BASE + "/search/q.php"

    def __init__(self, delay_range=(3, 6), max_notes=100):
        """
        Args:
            delay_range: 请求间隔范围（秒），随机化避免规律性
            max_notes: 最大抓取游记数（防止过度抓取）
        """
        self.delay_range = delay_range
        self.max_notes = max_notes
        self.notes = []

    def _fetch(self, url: str) -> str | None:
        """发起请求，返回 HTML 文本（带延迟和 UA 轮换）."""
        time.sleep(random.uniform(*self.delay_range))
        headers = random.choice(DEFAULT_HEADERS)
        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"  [warn] 请求失败 {url}: {e}")
            return None

    def search(self, keyword: str, pages: int = 3) -> list[dict]:
        """搜索游记，返回 [{note_id, url, title}].

        注：马蜂窝搜索结果页结构可能变化，选择器需根据实际 HTML 调整。
        """
        results = []
        for page in range(1, pages + 1):
            params = urllib.parse.urlencode({"q": keyword, "t": "notes", "p": page})
            url = f"{self.SEARCH_URL}?{params}"
            html = self._fetch(url)
            if not html:
                continue
            # 这里用简单的正则提取游记链接（实际应配合 BeautifulSoup）
            # 游记链接形如 /i/1234567.html 或 /notes/xxx.html
            pattern = re.compile(r'href="(?:https?://www\.mafengwo\.cn)?/(i|notes)/(\d+)\.html"[^>]*>([^<]+)')
            for match in pattern.finditer(html):
                note_id = match.group(2)
                title = match.group(3).strip()
                if note_id and title:
                    results.append({
                        "note_id": f"mafengwo_{note_id}",
                        "url": f"{self.BASE}/i/{note_id}.html",
                        "title": title,
                    })
        return results

    def extract_route(self, html: str) -> str:
        """从游记正文 HTML 中提取行程段落（地名序列）.

        启发式：找含"→"、"Day"、"第X天"等行程标记的段落。
        返回 "地名→地名→地名" 格式字符串，无法提取则返回空。
        """
        # 去掉 HTML 标签，保留文本
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)

        # 策略1：找 "→" 分隔的行程（马蜂窝游记常见格式）
        arrow_pattern = re.compile(r"([\u4e00-\u9fa5A-Za-z（）()]+(?:→[\u4e00-\u9fa5A-Za-z（）()]+){2,})")
        matches = arrow_pattern.findall(text)
        if matches:
            # 取最长的匹配
            longest = max(matches, key=len)
            return longest

        # 策略2：找 "Day X：景点A 景点B" 格式（备用）
        # ... 可扩展更多启发式
        return ""

    def crawl(self, keywords: list[str] = None) -> list[dict]:
        """完整爬取流程."""
        keywords = keywords or HARBIN_KEYWORDS
        all_results = []
        for kw in keywords:
            print(f"搜索: {kw}")
            results = self.search(kw, pages=2)
            print(f"  找到 {len(results)} 篇")
            all_results.extend(results)
            if len(all_results) >= self.max_notes:
                break

        # 去重（按 note_id）
        seen = set()
        unique = []
        for r in all_results:
            if r["note_id"] not in seen:
                seen.add(r["note_id"])
                unique.append(r)
        print(f"去重后: {len(unique)} 篇")

        # 抓取正文并提取路线
        routes = []
        for i, note in enumerate(unique[:self.max_notes]):
            print(f"  [{i+1}/{len(unique)}] {note['title'][:30]}")
            html = self._fetch(note["url"])
            if not html:
                continue
            route_text = self.extract_route(html)
            if route_text:
                routes.append({
                    "source": "mafengwo",
                    "title": note["title"],
                    "season": "",  # 可从正文提取季节关键词
                    "route": route_text,
                    "liked_count": 0,
                    "note_id": note["note_id"],
                })

        self.notes = routes
        print(f"\n提取到 {len(routes)} 条有效路线")
        return routes

    def save(self, out_path: str):
        """保存为 CSV（与 XHS 数据格式一致）."""
        if not self.notes:
            print("无数据可保存")
            return
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "title", "season",
                                                   "route", "liked_count", "note_id"])
            writer.writeheader()
            writer.writerows(self.notes)
        print(f"已保存 {len(self.notes)} 条到 {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="爬取马蜂窝哈尔滨游记")
    parser.add_argument("--max-notes", type=int, default=50)
    parser.add_argument("--output", default="data/raw/mafengwo_routes.csv")
    args = parser.parse_args()

    crawler = MafengwoCrawler(max_notes=args.max_notes)
    print("=" * 50)
    print("马蜂窝哈尔滨游记爬虫")
    print("⚠️  受反爬限制，可能需要人工验证或登录态")
    print("=" * 50)
    crawler.crawl()
    crawler.save(args.output)


if __name__ == "__main__":
    main()
