"""
Sequoia-X 选股策略：双指标合击当天收盘成交版 (CB_Combo_v6_Ultimate)
【跌破0轴纯金叉优化版】
"""

import numpy as np
import pandas as pd
from sequoia_x.strategy.base import BaseStrategy


class CbComboV6UltimateStrategy(BaseStrategy):
    """双指标合击策略：Squeeze 动量修复 + Vix 恐慌见底 (0轴下方纯金叉选股版)"""
    
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
        vixMult: float = 2.0,
        **kwargs
    ):
        super().__init__(engine, settings, **kwargs)
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
        
        fast_line_list = []
        for i in range(len(df)):
            if i < self.lengthKC:
                fast_line_list.append(0.0)
            else:
                fast_line_list.append(self._linear_regression_value(reg_source.iloc[:i+1], self.lengthKC))
                
        df['fast_line'] = fast_line_list
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
        
        for idx in range(len(df) - 10, len(df)):
            row_today = df.iloc[idx]
            row_yesterday = df.iloc[idx-1]
            
            gc_cross = (row_yesterday['fast_line'] <= row_yesterday['slow_line']) and (row_today['fast_line'] > row_today['slow_line'])
            vix_signal = row_today['is_vix_crit'] and not row_yesterday['is_vix_crit']
            
            if vix_signal:
                has_prepared = True
            
            if idx == len(df) - 1:
                if gc_cross and has_prepared and row_today['fast_line'] < 0:
                    return True
                    
            if gc_cross:
                has_prepared = False
                
        return False

    def run(self) -> list[str]:
        """
        实现基类要求的抽象方法，核心的选股循环调度
        """
        selected_symbols = []
        # 从父类继承的 self.engine 获取全市场股票代码
        all_symbols = self.engine.get_all_symbols()
        
        for symbol in all_symbols:
            try:
                # 从数据库读取单只股票的历史K线
                df = self.engine.get_kline(symbol)
                # 调用你写好的信号判断函数
                if self.check_signal(df):
                    selected_symbols.append(symbol)
            except Exception:
                # 某只股票计算出错时跳过，防止中断全市场扫描
                continue
                
        return selected_symbols
