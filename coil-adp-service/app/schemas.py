"""HTTP 请求/响应模型与字段级校验（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .psychrometrics import MoistAirState


# ---------------------------------------------------------------- 请求模型
class StateInput(BaseModel):
    """湿空气状态：干球温度 + 气压 + 一种湿度表示（三者择一）。"""

    model_config = ConfigDict(extra="forbid")

    t_db_c: float = Field(..., ge=-60.0, le=100.0, description="干球温度 [°C]")
    pressure_kpa: float = Field(101.325, gt=20.0, le=150.0, description="大气压力 [kPa]")
    rh: float | None = Field(None, gt=0.0, le=1.0, description="相对湿度 (0, 1]")
    dew_point_c: float | None = Field(None, ge=-80.0, le=100.0, description="露点温度 [°C]")
    humidity_ratio: float | None = Field(
        None, gt=0.0, description="含湿量 [kg/kg_da]"
    )

    @model_validator(mode="after")
    def _exactly_one_humidity(self) -> "StateInput":
        n = sum(
            v is not None for v in (self.rh, self.dew_point_c, self.humidity_ratio)
        )
        if n != 1:
            raise ValueError(
                "exactly one of rh / dew_point_c / humidity_ratio must be provided"
            )
        return self


class TargetInput(BaseModel):
    """目标出风条件：温度、湿度（三者择一）、SHR 的任意组合。"""

    model_config = ConfigDict(extra="forbid")

    t_db_c: float | None = Field(None, ge=-60.0, le=100.0, description="目标出风温度 [°C]")
    rh: float | None = Field(None, gt=0.0, le=1.0, description="目标出风相对湿度 (0, 1]")
    dew_point_c: float | None = Field(None, ge=-80.0, le=100.0, description="目标出风露点 [°C]")
    humidity_ratio: float | None = Field(
        None, gt=0.0, description="目标出风含湿量 [kg/kg_da]"
    )
    shr: float | None = Field(None, gt=0.0, le=1.0, description="目标显热比 (0, 1]")

    @model_validator(mode="after")
    def _at_most_one_humidity(self) -> "TargetInput":
        n = sum(
            v is not None for v in (self.rh, self.dew_point_c, self.humidity_ratio)
        )
        if n > 1:
            raise ValueError(
                "at most one of rh / dew_point_c / humidity_ratio may be provided"
            )
        return self

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (self.t_db_c, self.rh, self.dew_point_c, self.humidity_ratio, self.shr)
        )


class CoilSolveRequest(BaseModel):
    """盘管核算请求：进口状态 + 条件组合（见 README 的支持组合表）。"""

    model_config = ConfigDict(extra="forbid")

    inlet: StateInput
    bypass_factor: float | None = Field(None, ge=0.0, le=1.0, description="旁通系数 [0, 1]")
    adp_c: float | None = Field(None, ge=-60.0, le=100.0, description="装置露点 [°C]")
    target: TargetInput | None = None
    mass_flow_da_kg_s: float = Field(1.0, gt=0.0, description="干空气质量流量 [kg_da/s]")


class CoilAnalyzeRequest(BaseModel):
    """只反推 SHR 与冷量的请求：一组进口状态 + 一组出口状态。"""

    model_config = ConfigDict(extra="forbid")

    inlet: StateInput
    outlet: StateInput
    mass_flow_da_kg_s: float = Field(1.0, gt=0.0, description="干空气质量流量 [kg_da/s]")


# ---------------------------------------------------------------- 响应模型
class MoistAirStateOut(BaseModel):
    t_db_c: float
    t_wb_c: float
    t_dp_c: float
    rh: float
    humidity_ratio_kg_kg_da: float
    enthalpy_kj_kg_da: float
    vapor_pressure_pa: float
    saturation_vapor_pressure_pa: float
    specific_volume_m3_kg_da: float
    pressure_kpa: float

    @classmethod
    def from_state(cls, s: MoistAirState) -> "MoistAirStateOut":
        return cls(
            t_db_c=s.t_db_c,
            t_wb_c=s.wet_bulb_c,
            t_dp_c=s.dew_point_c,
            rh=s.rh,
            humidity_ratio_kg_kg_da=s.humidity_ratio,
            enthalpy_kj_kg_da=s.enthalpy_kj_kg_da,
            vapor_pressure_pa=s.vapor_pressure_pa,
            saturation_vapor_pressure_pa=s.saturation_vapor_pressure_pa,
            specific_volume_m3_kg_da=s.specific_volume_m3_kg_da,
            pressure_kpa=s.pressure_pa / 1000.0,
        )


class CapacityOut(BaseModel):
    mass_flow_da_kg_s: float
    q_total_kw: float
    q_sensible_kw: float
    q_latent_kw: float
    shr: float | None  # 总冷量为零（BF=1）时无定义


class CoilSolveResponse(BaseModel):
    mode: str
    inlet: MoistAirStateOut
    outlet: MoistAirStateOut
    adp_c: float
    adp_humidity_ratio_kg_kg_da: float
    bypass_factor: float
    capacity: CapacityOut


class CoilAnalyzeResponse(BaseModel):
    inlet: MoistAirStateOut
    outlet: MoistAirStateOut
    capacity: CapacityOut


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}


class ErrorResponse(BaseModel):
    error: ErrorBody
