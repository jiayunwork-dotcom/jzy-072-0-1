"""湿空气状态计算 —— 全服务唯一权威实现。

约定：
- 温度 °C，压力 Pa（HTTP 层用 kPa，进入本模块前换算），焓 kJ/kg_da，
  含湿量 kg/kg_da。
- 饱和蒸汽压只用 Magnus 公式，系数固定于本文件；露点、湿球等所有
  反算也都基于同一组公式，任何模块不得另抄一份。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InvalidHumiditySpecError, SupersaturatedStateError

# ---------------------------------------------------------------- 常数（固定，勿改）
MAGNUS_A = 17.625          # Magnus 公式系数
MAGNUS_B_C = 243.04        # Magnus 公式系数 [°C]
MAGNUS_P0_PA = 610.94      # Magnus 公式系数 [Pa]

CP_DA_KJ_KGK = 1.006       # 干空气定压比热 [kJ/(kg_da·K)]，显热量按它算
CP_WV_KJ_KGK = 1.86        # 水蒸气定压比热 [kJ/(kg·K)]
H_FG0_KJ_KG = 2501.0       # 0 °C 时水的汽化潜热 [kJ/kg]
MW_RATIO = 0.621945        # 水蒸气/干空气分子量之比
R_DA_J_KGK = 287.042       # 干空气气体常数 [J/(kg_da·K)]

RH_SUPER_SAT_TOLERANCE = 1e-9   # 相对湿度判据的浮点容差


# ---------------------------------------------------------------- 基本公式
def saturation_vapor_pressure_pa(t_db_c: float) -> float:
    """Magnus 公式：过水面饱和蒸汽压 [Pa]。"""
    return MAGNUS_P0_PA * math.exp(MAGNUS_A * t_db_c / (t_db_c + MAGNUS_B_C))


def dew_point_from_vapor_pressure_c(p_w_pa: float) -> float:
    """Magnus 反解露点 [°C]（与正算同一公式，保证互推一致）。"""
    if p_w_pa <= 0.0:
        raise InvalidHumiditySpecError(
            "vapor pressure must be positive to derive a dew point",
            {"p_w_pa": p_w_pa},
        )
    gamma = math.log(p_w_pa / MAGNUS_P0_PA)
    return MAGNUS_B_C * gamma / (MAGNUS_A - gamma)


def humidity_ratio_from_vapor_pressure(p_w_pa: float, p_pa: float) -> float:
    """由水蒸气分压力求含湿量 W = 0.621945·p_w/(p − p_w)。"""
    if p_w_pa >= p_pa:
        raise SupersaturatedStateError(
            "vapor pressure meets or exceeds atmospheric pressure; "
            "the state lies beyond the saturation limit",
            {"p_w_pa": p_w_pa, "pressure_pa": p_pa},
        )
    return MW_RATIO * p_w_pa / (p_pa - p_w_pa)


def vapor_pressure_from_humidity_ratio(w: float, p_pa: float) -> float:
    """由含湿量反求水蒸气分压力。"""
    return p_pa * w / (MW_RATIO + w)


def saturation_humidity_ratio(t_db_c: float, p_pa: float) -> float:
    """饱和含湿量 W_s(t, p)。"""
    return humidity_ratio_from_vapor_pressure(saturation_vapor_pressure_pa(t_db_c), p_pa)


def enthalpy_kj_kg_da(t_db_c: float, w: float) -> float:
    """h = cp_da·t + W·(h_fg0 + cp_wv·t)  [kJ/kg_da]。"""
    return CP_DA_KJ_KGK * t_db_c + w * (H_FG0_KJ_KG + CP_WV_KJ_KGK * t_db_c)


def temperature_from_enthalpy_c(h_kj_kg_da: float, w: float) -> float:
    """由 (h, W) 反算干球温度 —— 出风温度只能从这里来，不许单独算术平均。"""
    return (h_kj_kg_da - H_FG0_KJ_KG * w) / (CP_DA_KJ_KGK + CP_WV_KJ_KGK * w)


def relative_humidity(w: float, t_db_c: float, p_pa: float) -> float:
    return vapor_pressure_from_humidity_ratio(w, p_pa) / saturation_vapor_pressure_pa(t_db_c)


def specific_volume_m3_kg_da(t_db_c: float, w: float, p_pa: float) -> float:
    return R_DA_J_KGK * (t_db_c + 273.15) * (1.0 + 1.6078 * w) / p_pa


# ---------------------------------------------------------------- 湿球温度
def humidity_ratio_from_wet_bulb(t_db_c: float, t_wb_c: float, p_pa: float) -> float:
    """ASHRAE 湿球方程（过水面）：由干球/湿球温度反算含湿量。"""
    w_s_wb = saturation_humidity_ratio(t_wb_c, p_pa)
    return (
        (H_FG0_KJ_KG - 2.326 * t_wb_c) * w_s_wb - CP_DA_KJ_KGK * (t_db_c - t_wb_c)
    ) / (H_FG0_KJ_KG + CP_WV_KJ_KGK * t_db_c - 4.186 * t_wb_c)


def wet_bulb_temperature_c(t_db_c: float, w: float, p_pa: float) -> float:
    """二分法求湿球温度。

    热力学关系保证 t_dp <= t_wb <= t_db，区间两端符号相反，必收敛。
    """
    t_dp = dew_point_from_vapor_pressure_c(vapor_pressure_from_humidity_ratio(w, p_pa))
    lo, hi = t_dp, t_db_c
    if hi - lo <= 1e-12:
        return t_db_c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if humidity_ratio_from_wet_bulb(t_db_c, mid, p_pa) > w:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- 状态展开
@dataclass(frozen=True)
class MoistAirState:
    """统一展开后的湿空气状态（每 kg 干空气基准）。"""

    t_db_c: float
    pressure_pa: float
    humidity_ratio: float               # kg/kg_da
    rh: float                           # 相对湿度 (0,1]
    dew_point_c: float
    wet_bulb_c: float
    enthalpy_kj_kg_da: float
    vapor_pressure_pa: float
    saturation_vapor_pressure_pa: float
    specific_volume_m3_kg_da: float


def expand_state(
    t_db_c: float,
    pressure_pa: float,
    *,
    rh: float | None = None,
    dew_point_c: float | None = None,
    humidity_ratio: float | None = None,
) -> MoistAirState:
    """把「干球温度 + 气压 + 一种湿度表示」展开为完整湿空气状态。

    湿度三种表示（相对湿度 / 露点 / 含湿量）必须且只能给一种；
    展开出的含湿量超过饱和上限（RH > 1）时抛出 SupersaturatedStateError。
    """
    given = sum(x is not None for x in (rh, dew_point_c, humidity_ratio))
    if given != 1:
        raise InvalidHumiditySpecError(
            "exactly one of rh / dew_point_c / humidity_ratio must be provided",
            {"given": given},
        )
    if pressure_pa <= 0.0:
        raise InvalidHumiditySpecError("pressure must be positive", {"pressure_pa": pressure_pa})

    p_ws = saturation_vapor_pressure_pa(t_db_c)
    if rh is not None:
        if rh <= 0.0:
            raise InvalidHumiditySpecError("rh must be positive", {"rh": rh})
        p_w = rh * p_ws
    elif dew_point_c is not None:
        p_w = saturation_vapor_pressure_pa(dew_point_c)
    else:
        assert humidity_ratio is not None
        if humidity_ratio <= 0.0:
            raise InvalidHumiditySpecError(
                "humidity_ratio must be positive", {"humidity_ratio": humidity_ratio}
            )
        p_w = vapor_pressure_from_humidity_ratio(humidity_ratio, pressure_pa)

    # p_w >= p（气压）时此处直接抛 SupersaturatedStateError
    w = humidity_ratio_from_vapor_pressure(p_w, pressure_pa)

    rh_actual = p_w / p_ws
    if rh_actual > 1.0 + RH_SUPER_SAT_TOLERANCE:
        raise SupersaturatedStateError(
            "humidity ratio exceeds the saturation limit (relative humidity > 1); "
            "this is not a normal operating state",
            {
                "rh": rh_actual,
                "t_db_c": t_db_c,
                "humidity_ratio_kg_kg_da": w,
                "saturation_humidity_ratio_kg_kg_da": humidity_ratio_from_vapor_pressure(
                    p_ws, pressure_pa
                )
                if p_ws < pressure_pa
                else None,
            },
        )

    return MoistAirState(
        t_db_c=t_db_c,
        pressure_pa=pressure_pa,
        humidity_ratio=w,
        rh=rh_actual,
        dew_point_c=dew_point_from_vapor_pressure_c(p_w),
        wet_bulb_c=wet_bulb_temperature_c(t_db_c, w, pressure_pa),
        enthalpy_kj_kg_da=enthalpy_kj_kg_da(t_db_c, w),
        vapor_pressure_pa=p_w,
        saturation_vapor_pressure_pa=p_ws,
        specific_volume_m3_kg_da=specific_volume_m3_kg_da(t_db_c, w, pressure_pa),
    )
