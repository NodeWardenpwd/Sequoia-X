"""
Sequoia-X 选股策略：双指标合击当天收盘成交版 (CB_Combo_v6_Ultimate)
【最新修改：左侧动能连续2天收缩接近0轴版】
"""

import numpy as np
import pandas as pd
from sequoia_x.strategy.base import BaseStrategy


class CbComboV6UltimateStrategy(BaseStrategy):
    """双指标合击策略：Squeeze 动量修复 + Vix 恐慌见底 (0轴下方差值连续2天缩小版)"""
    
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
        super().__init__(engine=engine, settings=settings)
        self.webhook_key = "default" 
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
        核心信号判定：
        1. 此前有威廉恐慌触发（has_prepared）
        2. 今天快线依然在0轴下方安全区
        3. 快慢线绝对差值（动能柱）连续 2 天变小
        4. 快线连续 2 天接近0轴（即快线不能在恶化下跌，必须是在往上走或横盘修复）
        """
        if df is None or len(df) < max(self.lengthKC, self.pd_vix, self.bbl_vix) + 10:
            return False

        # 统一将K线字段转为小写
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
        
        # 滚动计算线性回归快线
        fast_line_list = []
        for i in range(len(df)):
            if i < self.lengthKC:
                fast_line_list.append(0.0)
            else:
                fast_line_list.append(self._linear_regression_value(reg_source.iloc[:i+1], self.lengthKC))
                
        df['fast_line'] = fast_line_list
        df['slow_line'] = df['fast_line'].ewm(span=9, adjust=False).mean()
        
        # 计算快慢线的绝对差值（即动能柱的绝对高度）
        df['current_diff'] = (df['fast_line'] - df['slow_line']).abs()

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
        # 3. 状态机追踪与【连续2天缩小接近0】信号判定
        # ==========================================
        has_prepared = False
        
        # 回溯过去15天的状态
        for idx in range(len(df) - 15, len(df)):
            row_today = df.iloc[idx]
            row_yesterday = df.iloc[idx-1]
            row_2days_ago = df.iloc[idx-2]
            
            # 判断威廉恐慌触发
            vix_signal = row_today['is_vix_crit'] and not row_yesterday['is_vix_crit']
            if vix_signal:
                has_prepared = True
            
            # 一旦发生金叉，说明左侧筑底阶段结束，重置准备状态
            gc_cross = (row_yesterday['fast_line'] <= row_yesterday['slow_line']) and (row_today['fast_line'] > row_today['slow_line'])
            if gc_cross:
                has_prepared = False
                
            # 到了最新的一根K线（即今天收盘），判定是否满足你的硬核左侧买入条件
            if idx == len(df) - 1:
                if has_prepared and row_today['fast_line'] < 0:
                    
                    # 条件1：快慢线的绝对差值连续2天缩小
                    # 昨天比前天小，且今天比昨天还小
                    diff_shrunk_2days = (row_today['current_diff'] < row_yesterday['current_diff']) and \
                                        (row_yesterday['current_diff'] < row_2days_ago['current_diff'])
                    
                    # 条件2：快线在0轴以下，且连续2天往0轴逼近（即快线值越来越大，或者维持不跌）
                    # 昨天的快线 >= 前天，且今天的快线 >= 昨天
                    fast_moving_to_zero = (row_today['fast_line'] >= row_yesterday['fast_line']) and \
                                          (row_yesterday['fast_line'] >= row_2days_ago['fast_line'])
                    
                    if diff_shrunk_2days and fast_moving_to_zero:
                        return True
                        
        return False
