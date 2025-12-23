import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch

from pathlib import Path
import json
from json import JSONEncoder
import csv

from typing import Optional, List, Tuple, Dict, Any, Union

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config

# This line makes the class available directly from the package
from .data_loader import load_data2

import math  # Tensor 없이 가볍게 연산하기 위해 사용
class BaseLogger:
    """JSON 및 CSV 저장/로드를 위한 공통 메서드를 제공하는 베이스 클래스"""
    
    def _clean_val(self, val: Any, digits: int = 4) -> float:
        """Tensor, Numpy 등을 순수 float로 변환하고 반올림"""
        if hasattr(val, 'item'):
            val = val.item()
        elif hasattr(val, '__len__') and not isinstance(val, (str, list, tuple)):
             if len(val) == 1: val = val[0]
        
        try:
            return round(float(val), digits)
        except (ValueError, TypeError):
            return val

    @staticmethod
    def load_list(input_filepath: Path) -> List[Dict]:
        """JSON 파일 로드 (파일 없으면 빈 리스트)"""
        try:
            with input_filepath.open('r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _extract_scalar(self, val):
        """내부 계산용: 반올림 없이 순수 값만 추출"""
        if hasattr(val, 'item'): val = val.item()
        return float(val)

class alg_optimization(BaseLogger):
    def __init__(self, day: Union[str, int], cov_name: str, space_size: int, 
                 lr: float, params: List[float], time: float, rmsre: float = 0.0): 
        """
        params: Log-Phi 스케일(최적화 결과) 입력 -> 내부에서 Physical Scale로 자동 변환 저장
        """
        self.day = str(day)
        self.cov_name = str(cov_name)
        self.space_size = self._clean_val(space_size, digits=0)
        self.lr = self._clean_val(lr, digits=1)
        
        # --- Log-Phi Scale -> Physical Scale 변환 로직 ---
        raw = [self._extract_scalar(p) for p in params]
        
        # 입력 순서: [log_phi1, log_phi2, log_phi3, log_phi4, adv_lat, adv_lon, log_nugget, loss]
        # (혹시 loss가 포함 안 된 7개라면 에러 방지 처리 필요하지만, 코드 흐름상 loss 포함됨)
        log_phi1, log_phi2, log_phi3, log_phi4 = raw[0], raw[1], raw[2], raw[3]
        adv_lat_raw, adv_lon_raw = raw[4], raw[5]
        log_nugget = raw[6]
        loss_val = raw[7] if len(raw) > 7 else 0.0
        
        phi1, phi2 = math.exp(log_phi1), math.exp(log_phi2)
        phi3, phi4 = math.exp(log_phi3), math.exp(log_phi4)
        
        calc_range_lon  = 1.0 / phi2
        calc_sigma_sq   = phi1 / phi2
        calc_range_lat  = calc_range_lon / math.sqrt(phi3)
        calc_range_time = calc_range_lon / math.sqrt(phi4) 
        calc_nugget     = math.exp(log_nugget)
        
        # 저장 (소수 4째 자리)
        self.sigma      = self._clean_val(calc_sigma_sq, digits=4)
        self.range_lat  = self._clean_val(calc_range_lat, digits=4)
        self.range_lon  = self._clean_val(calc_range_lon, digits=4)
        self.range_time = self._clean_val(calc_range_time, digits=4)
        self.advec_lat  = self._clean_val(adv_lat_raw, digits=4)
        self.advec_lon  = self._clean_val(adv_lon_raw, digits=4)
        self.nugget     = self._clean_val(calc_nugget, digits=4)
        self.loss       = self._clean_val(loss_val, digits=4)
        
        self.time       = self._clean_val(time, digits=4)
        self.rmsre      = self._clean_val(rmsre, digits=4)