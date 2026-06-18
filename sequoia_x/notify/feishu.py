"""飞书通知模块：将选股结果通过 Webhook 推送至飞书群。"""

import json
from datetime import date

import requests

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class FeishuNotifier:
    """飞书 Webhook 推送器。

    根据策略的 webhook_key 路由到对应的飞书机器人。
    若 webhook_key 未在 Settings.strategy_webhooks 中配置，
    则 fallback 到 Settings.feishu_webhook_url。
    """

    # --- 新增：精准策略含义字典，用于实现飞书悬停气泡提示 ---
    STRATEGY_DESCRIPTIONS = {
        "MaVolumeStrategy": "均线成交量突破：量价齐升动量策略，监控价格均线向上突破伴随成交量爆量放大，捕获大资金强力扫货个股。",
        "TurtleTradeStrategy": "海龟交易法则：经典趋势追踪策略，监控价格突破过去固定周期最高点，强者恒强，顺势捕捉主升浪。",
        "LimitUpShakeoutStrategy": "涨停板洗盘震荡：强势股龙头回踩策略，寻找近期刚拉涨停但随后缩量洗盘的个股，在准备二次拉升的临界点抄底主力。",
        "UptrendLimitDownStrategy": "上升趋势大跌反弹：黄金坑选股法，监控长期大趋势向上但短期因恐慌情绪遭遇暴跌或跌停的错杀优质个股，博弈强反弹。",
        "RpsBreakoutStrategy": "股价相对强度RPS突破：欧奈尔基本面技术龙头策略，筛选一段时间内涨幅超越市场绝大多数股票、且出现图形突破的核心题材龙头。",
        "PrivatePlacementStrategy": "定向增发套利：事件驱动型基本面策略，监控发布定增公告后股价跌至定增价附近的个股，利用大股东和机构的成本线作为天然的安全边际进行套利。"
    }

    def __init__(self, settings: Settings) -> None:
        """
        初始化 FeishuNotifier。

        Args:
            settings: Settings 实例，提供 Webhook URL 配置。
        """
        self.settings = settings

    @staticmethod
    def _to_xueqiu_code(code: str) -> str:
        """将纯数字代码转为雪球格式：6开头→SH，4/8开头→BJ，其余→SZ。"""
        if code.startswith("6"):
            return f"SH{code}"
        elif code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    @staticmethod
    def _get_stock_names(symbols: list[str]) -> dict[str, str]:
        """通过 baostock 批量查询股票名称，返回 {code: name} 映射。"""
        import baostock as bs
        bs.login()
        mapping = {}
        for code in symbols:
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            res = bs.query_stock_basic(code=f"{prefix}.{code}")
            while res.next():
                row = res.get_row_data()
                mapping[code] = row[1]  # 第2个字段是股票名称
        bs.logout()
        return mapping

    def _build_card(self, symbols: list[str], strategy_name: str) -> dict:
        today = date.today().strftime("%Y-%m-%d")
        names = self._get_stock_names(symbols)

        links: list[str] = []
        for code in symbols:
            xq_code = self._to_xueqiu_code(code)
            name = names.get(code, xq_code)
            links.append(f"[{name}](https://xueqiu.com/S/{xq_code})")

        symbol_text = " ".join(links) if links else "（无选股结果）"

        # --- 新增：提取对应的策略悬停说明，如未匹配则使用通用说明 ---
        desc = self.STRATEGY_DESCRIPTIONS.get(
            strategy_name, 
            "Sequoia-X 量化选股内核策略，基于历史价量动态计算得出。"
        )
        
        # 飞书卡片标准悬停气泡语法：[显示文本](? "悬停提示内容")
        hover_strategy_text = f"[{strategy_name} ❓](? \"{desc}\")"

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📈 Sequoia-X 选股播报 | {strategy_name}",
                    },
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            # 这里将原本的 {strategy_name} 替换成了带有悬停特性的 hover_strategy_text
                            "content": f"**日期：** {today}\n**策略：** {hover_strategy_text}\n**选股数量：** {len(symbols)}",
                        },
                    },
                    {"tag": "hr"},
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**选股列表：**\n{symbol_text}",
                        },
                    },
                ],
            },
        }

    def send(
        self,
        symbols: list[str],
        strategy_name: str,
        webhook_key: str = "default",
    ) -> None:
        """
        将选股结果格式化为飞书卡片消息并 POST 至对应 Webhook。

        根据 webhook_key 从 Settings 中查找专属 URL；
        若未配置，则 fallback 到 feishu_webhook_url。

        Args:
            symbols: 选股结果代码列表。
            strategy_name: 策略名称，用于卡片标题。
            webhook_key: 策略标识，用于路由到对应飞书机器人。

        Raises:
            不抛出异常，HTTP 失败时记录 ERROR 日志。
        """
        url = self.settings.get_webhook_url(webhook_key)
        payload = self._build_card(symbols, strategy_name)

        try:
            resp = requests.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            # 解析飞书真正的返回体
            resp_json = resp.json()

            # 飞书真正的成功标志是内部的 code == 0
            if resp.status_code != 200 or resp_json.get("code") != 0:
                logger.error(
                    f"飞书推送失败 [{webhook_key}] "
                    f"HTTP状态={resp.status_code} 飞书响应={resp.text}"
                )
            else:
                logger.info(f"飞书推送成功 [{webhook_key}]，共 {len(symbols)} 只股票")

        except requests.RequestException as exc:
            logger.error(f"飞书推送请求异常 [{webhook_key}]：{exc}")
