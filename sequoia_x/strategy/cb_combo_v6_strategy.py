"""
Sequoia-X 选股策略：双指标合击当天收盘成交版 (CB_Combo_v6_Ultimate)
【跌破0轴纯金叉优化版】
"""

import numpy as np
import pandas as pd


class CbComboV6UltimateStrategy:
    """双指标合击策略：Squeeze 动量修复 + Vix 恐慌见底 (0轴下方纯金叉选股版)"""
    
    def __init__(
        self, 
        lengthKC: int = 20, 
        multKC: float = 1.5, 
        lengthBB: int = 20, 
        multBB: float = 2.0,
        pd_vix: int = 22, 
        bbl_vix: int = 20, 
        vixMult: float = 2.0
    ):
        self.lengthKC = lengthKC
        self.multKC = multKC
        self.lengthBB = lengthBB
        self.multBB = multBB
        self.pd_vix = pd_vix
        self.bbl_vix = bbl_vix
        self.vixMult = vixMult

    def _linear_regression_value(self, series: pd.Series, length: int) -> float:
        """复刻 TradingView 中的 ta.linreg(source, length, 0) 线性回归预测值"""
        if len(series) < length:
            return 0.0
        y = series.tail(length).values
        x = np.arange(length)
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (length - 1) + intercept

    def check_signal(self, df: pd.DataFrame) -> bool:
        """
        扫描单只股票的历史K线数据
        返回当天是否满足：0轴以下金叉 + 伴随威廉恐慌见底信号
        """
        if df is None or len(df) < max(self.lengthKC, self.pd_vix, self.bbl_vix) + 10:
            return False

        # 统一将K线字段转为小写，防止报错
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ==========================================
        # 1. Squeeze Momentum 快慢线核心计算
        # ==========================================
        highest_high_kc = high.rolling(window=self.lengthKC).max()
        lowest_low_kc = low.rolling(window=self.lengthKC).min()
        sma_close_kc = close.rolling(window=self.lengthKC).mean()
        
        custom_avg = ((highest_high_kc + lowest_low_kc) / 2.0 + sma_close_kc) / 2.0
        reg_source = close - custom_avg
        
        # 滚动计算线性回归快线 (fastLine)
        fast_line_list = []
        for i in range(len(df)):
            if i < self.lengthKC:
                fast_line_list.append(0.0)
            else:
                fast_line_list.append(self._linear_regression_value(reg_source.iloc[:i+1], self.lengthKC))
                
        df['fast_line'] = fast_line_list
        # 慢线 = ta.ema(快线, 9)
        df['slow_line'] = df['fast_line'].ewm(span=9, adjust=False).mean()

        # ==========================================
        # 2. Williams Vix Fix 恐慌指标计算
        # ==========================================
        highest_close_vix = close.rolling(window=self.pd_vix).max()
        df['wvf'] = ((highest_close_vix - low) / highest_close_vix) * 100.0
        
        vix_mid = df['wvf'].rolling(window=self.bbl_vix).mean()
        vix_sdev = df['wvf'].rolling(window=self.bbl_vix).std(ddof=0)
        vix_upper = vix_mid + self.vixMult * vix_sdev
        df['is_vix_crit'] = df['wvf'] >= vix_upper

        # ==========================================
        # 3. 状态机追踪与纯金叉信号判定
        # ==========================================
        has_prepared = False
        
        # 仅对最近的K线进行状态回溯，确保选股的即时性
        for idx in range(len(df) - 10, len(df)):
            row_today = df.iloc[idx]
            row_yesterday = df.iloc[idx-1]
            
            # 判断今天是否金叉：昨天快线 <= 慢线，今天快线 > 慢线
            gc_cross = (row_yesterday['fast_line'] <= row_yesterday['slow_line']) and (row_today['fast_line'] > row_today['slow_line'])
            
            # 判断威廉恐慌触发（刚冲破布林上轨）
            vix_signal = row_today['is_vix_crit'] and not row_yesterday['is_vix_crit']
            
            if vix_signal:
                has_prepared = True
            
            # 到了最新的一根K线（即今天收盘），判定是否符合选股条件推送
            if idx == len(df) - 1:
                # 核心逻辑：必须是今天刚金叉、且快线在0轴以下安全区、且此前有恐慌砸盘的基础
                if gc_cross and has_prepared and row_today['fast_line'] < 0:
                    return True
                    
            if gc_cross:
                # 一旦发生金叉，无论是否完全符合，重置准备状态
                has_prepared = False
                
        return False
