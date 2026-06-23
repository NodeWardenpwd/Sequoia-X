"""
Sequoia-X 选股策略：双指标合击当天收盘成交版 (CB_Combo_v6_Ultimate)
【完美修复：加入底座必需的 run 函数 + 连续2天差值缩小逼近0轴版】
"""

import numpy as np
import pandas as pd
from sequoia_x.strategy.base import BaseStrategy


class CbComboV6UltimateStrategy(BaseStrategy):
    """双指标合击策略：Squeeze 动量修复 + Vix 恐慌见底 (完全融入框架 + 0轴下方差值连续2天缩小)"""
    
    def __init__(
        self, 
        engine, 
        settings,
        lengthKC: int = 20, 
        multKC: float = 1.5, 
        lengthBB: int = 20, 
        multBB: float = 2.0,
        pd_vix: int = 22, 
        bbl_vix: int = 20, 
        vixMult: float = 2.0
    ):
        # 1. 初始化父类底座
        super().__init__(engine=engine, settings=settings)
        # 对齐飞书推送路由关键字
        self.webhook_key = "default" 
        
        # 2. 绑定参数
        self.lengthKC = lengthKC
        self.multKC = multKC
        self.lengthBB = lengthBB
        self.multBB = multBB
        self.pd_vix = pd_vix
        self.bbl_vix = bbl_vix
        self.vixMult = vixMult

    def _linear_regression_value(self, series: pd.Series, length: int) -> float:
        """复刻 TradingView 中的 ta.linreg 线性回归预测值"""
        if len(series) < length:
            return 0.0
        y = series.tail(length).values
        x = np.arange(length)
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (length - 1) + intercept

    def check_signal(self, df: pd.DataFrame) -> bool:
        """
        核心信号判定：连续 2 天动能差值缩小且快线逼近 0 轴
        """
        if df is None or len(df) < max(self.lengthKC, self.pd_vix, self.bbl_vix) + 10:
            return False

        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 1. Squeeze Momentum 计算
        highest_high_kc = high.rolling(window=self.lengthKC).max()
        lowest_low_kc = low.rolling(window=self.lengthKC).min()
        sma_close_kc = close.rolling(window=self.lengthKC).mean()
        
        custom_avg = ((highest_high_kc + lowest_low_kc) / 2.0 + sma_close_kc) / 2.0
        reg_source = close - custom_avg
        
        fast_line_list = []
        for i in range(len(df)):
            if i < self.lengthKC:
                fast_line_list.append(0.0)
            else:
                fast_line_list.append(self._linear_regression_value(reg_source.iloc[:i+1], self.lengthKC))
                
        df['fast_line'] = fast_line_list
        df['slow_line'] = df['fast_line'].ewm(span=9, adjust=False).mean()
        df['current_diff'] = (df['fast_line'] - df['slow_line']).abs()

        # 2. Williams Vix Fix 计算
        highest_close_vix = close.rolling(window=self.pd_vix).max()
        df['wvf'] = ((highest_close_vix - low) / highest_close_vix) * 100.0
        
        vix_mid = df['wvf'].rolling(window=self.bbl_vix).mean()
        vix_sdev = df['wvf'].rolling(window=self.bbl_vix).std(ddof=0)
        vix_upper = vix_mid + self.vixMult * vix_sdev
        df['is_vix_crit'] = df['wvf'] >= vix_upper

        # 3. 状态判定
        has_prepared = False
        for idx in range(len(df) - 15, len(df)):
            row_today = df.iloc[idx]
            row_yesterday = df.iloc[idx-1]
            row_2days_ago = df.iloc[idx-2]
            
            vix_signal = row_today['is_vix_crit'] and not row_yesterday['is_vix_crit']
            if vix_signal:
                has_prepared = True
            
            gc_cross = (row_yesterday['fast_line'] <= row_yesterday['slow_line']) and (row_today['fast_line'] > row_today['slow_line'])
            if gc_cross:
                has_prepared = False
                
            if idx == len(df) - 1:
                if has_prepared and row_today['fast_line'] < 0:
                    # 动能连续2天缩小
                    diff_shrunk_2days = (row_today['current_diff'] < row_yesterday['current_diff']) and \
                                        (row_yesterday['current_diff'] < row_2days_ago['current_diff'])
                    # 快线连续2天往0轴靠拢
                    fast_moving_to_zero = (row_today['fast_line'] >= row_yesterday['fast_line']) and \
                                          (row_yesterday['fast_line'] >= row_2days_ago['fast_line'])
                    
                    if diff_shrunk_2days and fast_moving_to_zero:
                        return True
                        
        return False

    def run(self) -> list[str]:
        """
        【核心补全】：实现底座抽象方法，遍历全市场股票执行筛选
        """
        selected_symbols = []
        # 从底座引擎获取当前参与计算的所有股票代码
        all_symbols = self.engine.get_all_symbols()
        
        for symbol in all_symbols:
            try:
                # 获取单只股票的K线数据
                df = self.engine.get_kline(symbol)
                if df is not None and not df.empty:
                    # 传入 check_signal 进行核心数学计算
                    if self.check_signal(df):
                        selected_symbols.append(symbol)
            except Exception:
                continue
                
        return selected_symbols
