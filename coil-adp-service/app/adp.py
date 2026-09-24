"""装置露点（ADP）求解与旁通系数加权。

盘管模型：出风是「贴着 ADP 饱和的那部分风」与「原样绕过去的旁通风」
按同一个旁通系数 BF 的线性混合：

    W_out = W_adp + BF·(W_in − W_adp)
    h_out = h_adp + BF·(h_in − h_adp)

出口温度不单独算术平均，而是由 (h_out, W_out) 反算，保证温度、含湿量、
焓三者加权方式自洽。

支持的条件组合（其余组合一律报「条件不足/矛盾」，不瞎猜）：
    BF + ADP                  正算出风
    BF + 目标出风温度          反解 ADP
    BF + 目标 SHR              反解 ADP
    ADP + 目标出风温度         反解 BF
    目标出风温度 + 目标出风湿度  联立反解 ADP 与 BF
    目标出风温度 + 目标 SHR      联立反解 ADP 与 BF
"""
from __future__ import annotations

from dataclasses import dataclass

from .capacity import CapacityBreakdown, decompose_capacity
from .errors import (
    AdpConstraintError,
    BypassFactorError,
    ConflictingConditionsError,
    InsufficientConditionsError,
    NoSolutionError,
    ShrOutOfRangeError,
)
from .psychrometrics import (
    CP_DA_KJ_KGK,
    CP_WV_KJ_KGK,
    H_FG0_KJ_KG,
    MoistAirState,
    enthalpy_kj_kg_da,
    expand_state,
    saturation_humidity_ratio,
    temperature_from_enthalpy_c,
)

T_ADP_MIN_C = -60.0        # ADP 迭代区间下限，低于此温度对空调盘管无意义
_BISECTION_MAX_ITER = 200
_BRACKET_SHRINK_C = 1e-6   # 区间上端内缩，避开饱和边界上的平凡根/边界奇异
_BF_TOL = 1e-9
_SCAN_PROBES = 400         # 从区间上端向下扫描的探针数


# ---------------------------------------------------------------- 基础工具
def _sat_point(t_c: float, p_pa: float) -> tuple[float, float]:
    """饱和线上一点：(W_s, h_s)。"""
    w = saturation_humidity_ratio(t_c, p_pa)
    return w, enthalpy_kj_kg_da(t_c, w)


def _bisect(func, lo: float, hi: float, what: str) -> float:
    """在已包围根的区间内二分求根；调用方必须保证 func(lo)·func(hi) <= 0。"""
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo < 0.0) == (f_hi < 0.0):
        raise NoSolutionError(
            f"cannot bracket a root while solving {what}",
            {"lo_c": lo, "hi_c": hi, "f_lo": f_lo, "f_hi": f_hi},
        )
    for _ in range(_BISECTION_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0 or (hi - lo) < 1e-12:
            return mid
        if (f_lo < 0.0) == (f_mid < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    # 二分法 200 次必然压到机器精度；走到这里说明函数异常，按不收敛报错
    raise NoSolutionError(f"iteration did not converge while solving {what}")


def _find_root_from_top(func, lo: float, hi: float, what: str) -> float:
    """从区间上端向下扫描第一个变号区间，再二分精确化。

    饱和线（W_s, h_s）是凹曲线，进口→出口的连线可能与它有两个交点；
    物理上的 ADP 是沿「进口→出口」射线首先遇到的那个（最上方的）交点，
    因此必须从区间上端向下找第一个根。找不到即报迭代不收敛，
    绝不返回中间值。
    """
    if hi <= lo:
        raise NoSolutionError(
            f"empty search interval while solving {what}", {"lo_c": lo, "hi_c": hi}
        )
    f_hi = func(hi)
    if f_hi == 0.0:
        return hi
    step = (hi - lo) / _SCAN_PROBES
    t_prev, f_prev = hi, f_hi
    for k in range(1, _SCAN_PROBES + 1):
        t = hi - k * step
        f_t = func(t)
        if f_t == 0.0:
            return t
        if (f_prev < 0.0) != (f_t < 0.0):
            return _bisect(func, t, t_prev, what)
        t_prev, f_prev = t, f_t
    raise NoSolutionError(
        f"iteration did not converge: no root found while solving {what}; "
        "the target may be physically unreachable from the inlet state",
        {"lo_c": lo, "hi_c": hi},
    )


def validate_adp(inlet: MoistAirState, t_adp_c: float) -> None:
    """ADP 物理约束：低于进口湿球温度，且饱和含湿量不超过进口含湿量。

    违反任一条都说明这不是一个真在做冷却去湿的工况，直接报错。
    """
    if not t_adp_c < inlet.wet_bulb_c:
        raise AdpConstraintError(
            "apparatus dew point must be lower than the inlet wet-bulb temperature; "
            "this is not a cooling/dehumidifying coil process",
            {"t_adp_c": t_adp_c, "t_wb_in_c": inlet.wet_bulb_c},
        )
    w_adp = saturation_humidity_ratio(t_adp_c, inlet.pressure_pa)
    if w_adp > inlet.humidity_ratio * (1.0 + 1e-12):
        raise AdpConstraintError(
            "ADP humidity ratio exceeds the inlet humidity ratio; the process "
            "would humidify rather than dehumidify",
            {
                "t_adp_c": t_adp_c,
                "w_adp_kg_kg_da": w_adp,
                "w_in_kg_kg_da": inlet.humidity_ratio,
            },
        )


def _require_valid_bf(bf: float, context: str) -> float:
    if not -_BF_TOL <= bf <= 1.0 + _BF_TOL:
        raise NoSolutionError(
            f"target is not reachable ({context}): computed bypass factor "
            f"{bf:.6g} lies outside [0, 1]",
            {"bypass_factor": bf, "context": context},
        )
    return min(1.0, max(0.0, bf))


def _mix(inlet: MoistAirState, t_adp_c: float, bf: float) -> tuple[float, float, float]:
    """纯计算的旁通混合（不做校验），返回 (t_out, w_out, h_out)。"""
    w_adp, h_adp = _sat_point(t_adp_c, inlet.pressure_pa)
    w_out = w_adp + bf * (inlet.humidity_ratio - w_adp)
    h_out = h_adp + bf * (inlet.enthalpy_kj_kg_da - h_adp)
    return temperature_from_enthalpy_c(h_out, w_out), w_out, h_out


# ---------------------------------------------------------------- 正算
def leaving_state(inlet: MoistAirState, t_adp_c: float, bf: float) -> MoistAirState:
    """正算出风：W 与 h 用同一个 BF 加权，出口温度由 (h_out, W_out) 反算。"""
    if not 0.0 - _BF_TOL <= bf <= 1.0 + _BF_TOL:
        raise BypassFactorError("bypass factor must lie in [0, 1]", {"bypass_factor": bf})
    bf = min(1.0, max(0.0, bf))
    validate_adp(inlet, t_adp_c)
    _, w_out, h_out = _mix(inlet, t_adp_c, bf)
    t_out = temperature_from_enthalpy_c(h_out, w_out)
    return expand_state(t_out, inlet.pressure_pa, humidity_ratio=w_out)


# ---------------------------------------------------------------- 反解单元
def _bf_for_target_t_out(inlet: MoistAirState, t_adp_c: float, t_out_c: float) -> float:
    """给定 ADP 与目标出风温度，闭式解 BF。

    h_out(BF) 与 W_out(BF) 都是 BF 的线性函数，约束
    h_out = cp_da·t_out + W_out·(h_fg0 + cp_wv·t_out) 关于 BF 是线性方程。
    """
    w_adp, h_adp = _sat_point(t_adp_c, inlet.pressure_pa)
    c = H_FG0_KJ_KG + CP_WV_KJ_KGK * t_out_c
    numerator = CP_DA_KJ_KGK * t_out_c + w_adp * c - h_adp
    denominator = (inlet.enthalpy_kj_kg_da - h_adp) - (inlet.humidity_ratio - w_adp) * c
    if denominator <= 0.0:
        raise NoSolutionError(
            "degenerate mixing line while solving the bypass factor",
            {"t_adp_c": t_adp_c, "t_out_c": t_out_c},
        )
    return numerator / denominator


def solve_bf_given_adp_t_out(inlet: MoistAirState, t_adp_c: float, t_out_c: float) -> float:
    """模式：ADP + 目标出风温度 → BF。"""
    validate_adp(inlet, t_adp_c)
    return _require_valid_bf(
        _bf_for_target_t_out(inlet, t_adp_c, t_out_c), "ADP + target outlet temperature"
    )


def solve_adp_given_bf_t_out(inlet: MoistAirState, bf: float, t_out_c: float) -> float:
    """模式：BF + 目标出风温度 → ADP。"""
    if t_out_c > inlet.t_db_c + 1e-12:
        raise NoSolutionError(
            "target outlet temperature is above the inlet temperature; "
            "not a cooling process",
            {"t_out_c": t_out_c, "t_in_c": inlet.t_db_c},
        )
    if bf >= 1.0:
        raise ConflictingConditionsError(
            "bypass_factor = 1 means no heat transfer at all; the apparatus dew "
            "point cannot be identified from a target outlet temperature"
        )
    if bf == 0.0:
        # 出口即 ADP 饱和状态，ADP 就是目标出风温度
        validate_adp(inlet, t_out_c)
        return t_out_c
    hi = min(t_out_c, inlet.dew_point_c, inlet.wet_bulb_c) - _BRACKET_SHRINK_C
    t_adp = _find_root_from_top(
        lambda t: _bf_for_target_t_out(inlet, t, t_out_c) - bf,
        T_ADP_MIN_C,
        hi,
        "ADP from bypass factor and target outlet temperature",
    )
    validate_adp(inlet, t_adp)
    return t_adp


def solve_adp_given_bf_shr(inlet: MoistAirState, bf: float, shr: float) -> float:
    """模式：BF + 目标 SHR → ADP。"""
    if not 0.0 < shr <= 1.0:
        raise ShrOutOfRangeError("target SHR must lie in (0, 1]", {"shr": shr})
    if bf >= 1.0:
        raise ConflictingConditionsError(
            "bypass_factor = 1 means no heat transfer at all; SHR is undefined"
        )

    def shr_gap(t_adp_c: float) -> float:
        t_out, _, h_out = _mix(inlet, t_adp_c, bf)
        d_h = inlet.enthalpy_kj_kg_da - h_out
        if d_h <= 0.0:
            return float("inf")
        return CP_DA_KJ_KGK * (inlet.t_db_c - t_out) / d_h - shr

    hi = min(inlet.dew_point_c, inlet.wet_bulb_c) - _BRACKET_SHRINK_C
    t_adp = _find_root_from_top(
        shr_gap, T_ADP_MIN_C, hi, "ADP from bypass factor and target SHR"
    )
    validate_adp(inlet, t_adp)
    return t_adp


def solve_adp_bf_given_target_state(
    inlet: MoistAirState, t_out_c: float, w_out: float
) -> tuple[float, float]:
    """模式：目标出风全状态（温度 + 湿度）→ 联立反解 ADP 与 BF。

    进口→出口的连线与饱和线的交点即 ADP（沿射线方向首先遇到的交点），
    BF 由混合比例确定。
    """
    if w_out > inlet.humidity_ratio * (1.0 + 1e-12):
        raise NoSolutionError(
            "target outlet is more humid than the inlet; a cooling coil cannot humidify",
            {"w_out_kg_kg_da": w_out, "w_in_kg_kg_da": inlet.humidity_ratio},
        )
    if t_out_c > inlet.t_db_c + 1e-12:
        raise NoSolutionError(
            "target outlet temperature is above the inlet temperature; "
            "not a cooling process",
            {"t_out_c": t_out_c, "t_in_c": inlet.t_db_c},
        )
    h_out = enthalpy_kj_kg_da(t_out_c, w_out)
    d_w = inlet.humidity_ratio - w_out
    d_h = inlet.enthalpy_kj_kg_da - h_out

    if d_w <= 1e-9:
        # 近纯显热过程：等 W 线与饱和线交于进口露点
        t_adp = inlet.dew_point_c
        validate_adp(inlet, t_adp)
        _, h_adp = _sat_point(t_adp, inlet.pressure_pa)
        if abs(inlet.enthalpy_kj_kg_da - h_adp) < 1e-12:
            raise NoSolutionError(
                "degenerate target: outlet coincides with a saturated inlet"
            )
        bf = (h_out - h_adp) / (inlet.enthalpy_kj_kg_da - h_adp)
        return t_adp, _require_valid_bf(bf, "target outlet state")

    def collinearity(t_c: float) -> float:
        w_s, h_s = _sat_point(t_c, inlet.pressure_pa)
        return (inlet.enthalpy_kj_kg_da - h_s) * d_w - d_h * (inlet.humidity_ratio - w_s)

    hi = min(inlet.dew_point_c, inlet.wet_bulb_c) - _BRACKET_SHRINK_C
    t_adp = _find_root_from_top(
        collinearity, T_ADP_MIN_C, hi, "ADP from target outlet state"
    )
    validate_adp(inlet, t_adp)
    w_adp, h_adp = _sat_point(t_adp, inlet.pressure_pa)
    if abs(inlet.humidity_ratio - w_adp) > 1e-12:
        bf = (w_out - w_adp) / (inlet.humidity_ratio - w_adp)
    else:
        bf = (h_out - h_adp) / (inlet.enthalpy_kj_kg_da - h_adp)
    return t_adp, _require_valid_bf(bf, "target outlet state")


def solve_adp_bf_given_t_out_shr(
    inlet: MoistAirState, t_out_c: float, shr: float
) -> tuple[float, float]:
    """模式：目标出风温度 + 目标 SHR → 联立反解 ADP 与 BF。

    SHR 定义直接给出目标焓差 Δh = cp_da·ΔT/SHR，于是问题化为
    「找 ADP 使正算出的 h_out 等于目标值」，BF 由闭式公式给出。
    """
    if not 0.0 < shr <= 1.0:
        raise ShrOutOfRangeError("target SHR must lie in (0, 1]", {"shr": shr})
    if t_out_c > inlet.t_db_c + 1e-12:
        raise NoSolutionError(
            "target outlet temperature is above the inlet temperature; "
            "not a cooling process",
            {"t_out_c": t_out_c, "t_in_c": inlet.t_db_c},
        )
    delta_h = CP_DA_KJ_KGK * (inlet.t_db_c - t_out_c) / shr
    h_out_target = inlet.enthalpy_kj_kg_da - delta_h
    w_out_implied = (h_out_target - CP_DA_KJ_KGK * t_out_c) / (
        H_FG0_KJ_KG + CP_WV_KJ_KGK * t_out_c
    )
    if w_out_implied < -1e-12:
        raise NoSolutionError(
            "target SHR implies a negative outlet humidity ratio; physically unreachable",
            {"shr": shr, "t_out_c": t_out_c},
        )

    def h_gap(t_adp_c: float) -> float:
        bf = _bf_for_target_t_out(inlet, t_adp_c, t_out_c)
        _, _, h_out = _mix(inlet, t_adp_c, bf)
        return h_out - h_out_target

    hi = min(t_out_c, inlet.dew_point_c, inlet.wet_bulb_c) - _BRACKET_SHRINK_C
    t_adp = _find_root_from_top(
        h_gap, T_ADP_MIN_C, hi, "ADP from target outlet temperature and SHR"
    )
    validate_adp(inlet, t_adp)
    bf = _require_valid_bf(
        _bf_for_target_t_out(inlet, t_adp, t_out_c),
        "target outlet temperature + SHR",
    )
    return t_adp, bf


# ---------------------------------------------------------------- 模式分派
@dataclass(frozen=True)
class TargetSpec:
    """目标出风条件（湿度已在 HTTP 层换算为含湿量）。"""

    t_db_c: float | None = None
    w_kg_kg_da: float | None = None
    shr: float | None = None


@dataclass(frozen=True)
class CoilSolution:
    mode: str
    inlet: MoistAirState
    outlet: MoistAirState
    adp_c: float
    adp_humidity_ratio_kg_kg_da: float
    bypass_factor: float
    capacity: CapacityBreakdown


_SUPPORTED_SETS = (
    frozenset({"bf", "adp"}),
    frozenset({"bf", "t"}),
    frozenset({"bf", "shr"}),
    frozenset({"adp", "t"}),
    frozenset({"t", "w"}),
    frozenset({"t", "shr"}),
)

SUPPORTED_COMBINATIONS = (
    "bypass_factor + adp_c (forward calculation)",
    "bypass_factor + target.t_db_c",
    "bypass_factor + target.shr",
    "adp_c + target.t_db_c",
    "target.t_db_c + one target humidity field (rh / dew_point_c / humidity_ratio)",
    "target.t_db_c + target.shr",
)


def _raise_bad_combination(given: frozenset[str]) -> None:
    if any(s < given for s in _SUPPORTED_SETS):
        raise ConflictingConditionsError(
            "conditions are over-specified; supported combinations are: "
            + "; ".join(SUPPORTED_COMBINATIONS),
            {"given": sorted(given)},
        )
    raise InsufficientConditionsError(
        "conditions are insufficient to determine the apparatus dew point and "
        "bypass factor; supported combinations are: "
        + "; ".join(SUPPORTED_COMBINATIONS),
        {"given": sorted(given)},
    )


def solve_coil(
    inlet: MoistAirState,
    *,
    bypass_factor: float | None,
    adp_c: float | None,
    target: TargetSpec | None,
    mass_flow_da_kg_s: float,
) -> CoilSolution:
    """按给定条件组合分派求解，返回装置露点、旁通系数、出风状态与冷量。"""
    t_target = target.t_db_c if target else None
    w_target = target.w_kg_kg_da if target else None
    shr_target = target.shr if target else None

    given = frozenset(
        name
        for name, val in (
            ("bf", bypass_factor),
            ("adp", adp_c),
            ("t", t_target),
            ("w", w_target),
            ("shr", shr_target),
        )
        if val is not None
    )

    if given == frozenset({"bf", "adp"}):
        mode = "forward_bf_adp"
        t_adp, bf = adp_c, bypass_factor
    elif given == frozenset({"bf", "t"}):
        mode = "inverse_bf_target_t_out"
        bf = bypass_factor
        t_adp = solve_adp_given_bf_t_out(inlet, bf, t_target)
    elif given == frozenset({"bf", "shr"}):
        mode = "inverse_bf_target_shr"
        bf = bypass_factor
        t_adp = solve_adp_given_bf_shr(inlet, bf, shr_target)
    elif given == frozenset({"adp", "t"}):
        mode = "inverse_adp_target_t_out"
        t_adp = adp_c
        bf = solve_bf_given_adp_t_out(inlet, t_adp, t_target)
    elif given == frozenset({"t", "w"}):
        mode = "inverse_target_state"
        t_adp, bf = solve_adp_bf_given_target_state(inlet, t_target, w_target)
    elif given == frozenset({"t", "shr"}):
        mode = "inverse_target_t_out_shr"
        t_adp, bf = solve_adp_bf_given_t_out_shr(inlet, t_target, shr_target)
    else:
        _raise_bad_combination(given)

    outlet = leaving_state(inlet, t_adp, bf)
    capacity = decompose_capacity(inlet, outlet, mass_flow_da_kg_s, require_cooling=False)
    return CoilSolution(
        mode=mode,
        inlet=inlet,
        outlet=outlet,
        adp_c=t_adp,
        adp_humidity_ratio_kg_kg_da=saturation_humidity_ratio(t_adp, inlet.pressure_pa),
        bypass_factor=bf,
        capacity=capacity,
    )
