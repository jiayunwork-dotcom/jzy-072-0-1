"""冷量分解：总冷量 = 显热 + 潜热，并校验显热比 SHR 自洽。

约定（按需求固定）：
- 显热量 Q_s = m_dot_da · cp_da · (T_in − T_out)，cp_da 取干空气比热 1.006 kJ/(kg·K)；
- 总冷量 Q_t = m_dot_da · (h_in − h_out)；
- 潜热量 Q_l = Q_t − Q_s；
三者对同一组进出口状态必然自洽（Q_s + Q_l == Q_t 恒成立）。
SHR = Q_s / Q_t 必须落在 (0, 1]，否则拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import NotCoolingProcessError, ShrOutOfRangeError
from .psychrometrics import CP_DA_KJ_KGK, MoistAirState

# 总冷量低于该阈值视为零（对应 BF=1 无换热），SHR 无定义
_Q_ZERO_EPS_KW = 1e-9
_SHR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CapacityBreakdown:
    mass_flow_da_kg_s: float
    q_total_kw: float
    q_sensible_kw: float
    q_latent_kw: float
    shr: float | None  # 总冷量为零（BF=1）时无定义，置 None


def decompose_capacity(
    inlet: MoistAirState,
    outlet: MoistAirState,
    mass_flow_da_kg_s: float,
    *,
    require_cooling: bool,
) -> CapacityBreakdown:
    """对同一组进出口状态分解冷量。

    require_cooling=True（进出口校核接口）时，总冷量必须为正，
    否则说明不是冷却过程；SHR 落在 (0, 1] 之外一律拒绝。
    """
    q_total = mass_flow_da_kg_s * (inlet.enthalpy_kj_kg_da - outlet.enthalpy_kj_kg_da)
    q_sensible = mass_flow_da_kg_s * CP_DA_KJ_KGK * (inlet.t_db_c - outlet.t_db_c)
    q_latent = q_total - q_sensible

    if abs(q_total) < _Q_ZERO_EPS_KW:
        if require_cooling:
            raise NotCoolingProcessError(
                "total capacity is not positive: this is not a cooling process",
                {"q_total_kw": q_total},
            )
        shr = None
    else:
        if require_cooling and q_total < 0.0:
            raise NotCoolingProcessError(
                "total capacity is negative: outlet enthalpy exceeds inlet enthalpy, "
                "this is heating rather than cooling",
                {"q_total_kw": q_total},
            )
        shr = q_sensible / q_total

    if shr is not None and not (0.0 < shr <= 1.0 + _SHR_TOLERANCE):
        raise ShrOutOfRangeError(
            "sensible heat ratio falls outside (0, 1]; the given states are not a "
            "cooling/dehumidifying coil process (e.g. outlet may be wetter than inlet)",
            {
                "shr": shr,
                "q_sensible_kw": q_sensible,
                "q_latent_kw": q_latent,
                "q_total_kw": q_total,
            },
        )

    return CapacityBreakdown(
        mass_flow_da_kg_s=mass_flow_da_kg_s,
        q_total_kw=q_total,
        q_sensible_kw=q_sensible,
        q_latent_kw=q_latent_kw,
        shr=shr,
    )
