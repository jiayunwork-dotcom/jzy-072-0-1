"""领域错误定义。

所有「这不是一个正常冷却去湿工况 / 条件不够 / 迭代不收敛」的拒绝，
都以结构化错误抛出，由 HTTP 层统一序列化为：

    {"error": {"code": ..., "message": ..., "details": {...}}}
"""
from __future__ import annotations


class DomainError(Exception):
    """业务/物理约束错误基类。"""

    code: str = "DOMAIN_ERROR"
    status_code: int = 422

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidHumiditySpecError(DomainError):
    """湿度表示方式不合法（缺失、冲突或超出物理意义）。"""

    code = "INVALID_HUMIDITY_SPEC"


class SupersaturatedStateError(DomainError):
    """含湿量超过饱和上限（相对湿度 > 1），不能当正常工况处理。"""

    code = "SUPER_SATURATED_STATE"


class AdpConstraintError(DomainError):
    """装置露点违反物理约束：不低于进口湿球，或其含湿量超过进口含湿量。"""

    code = "ADP_CONSTRAINT_VIOLATION"


class NoSolutionError(DomainError):
    """反算迭代不收敛 / 物理上无解。绝不把中间值当结果返回。"""

    code = "NO_PHYSICAL_SOLUTION"


class InsufficientConditionsError(DomainError):
    """给定条件不足以唯一确定装置露点与旁通系数。"""

    code = "INSUFFICIENT_CONDITIONS"


class ConflictingConditionsError(DomainError):
    """条件过约束或互相矛盾。"""

    code = "CONFLICTING_CONDITIONS"


class BypassFactorError(DomainError):
    """旁通系数落在 [0, 1] 之外。"""

    code = "BYPASS_FACTOR_OUT_OF_RANGE"


class NotCoolingProcessError(DomainError):
    """总冷量非正：这组进出口状态根本不是冷却过程。"""

    code = "NOT_A_COOLING_PROCESS"


class ShrOutOfRangeError(DomainError):
    """显热比落在 (0, 1] 之外。"""

    code = "SHR_OUT_OF_RANGE"
