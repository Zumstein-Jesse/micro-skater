# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""XL330 servo heat model (skate student, 2026-11-06).

Electrical loss -> two-node thermal network per servo:

    P_j   = I_j^2 * R0 * (1 + ALPHA_CU * (T_w - T_REF))        copper loss of servo j (W)
    C_w dT_w/dt = P_j - (T_w - T_h) / R_wh                      winding node
    C_h dT_h/dt = (T_w - T_h) / R_wh - (T_h - T_amb) / R_ha + P_idle   housing / PCB node

I_j is the motor current of the BAM M6 XL330 model the repo trains with (bam/params/xl330/m6.json): the actuator
returns tau = kt * I (output referred, kt = 0.366 N m/A, R0 = 2.81 ohm fitted; datasheet XL330-M288 at 5 V: stall
0.52 N m / 1.47 A -> kt ~ 0.354 N m/A, R ~ 3.4 ohm, so the fit is consistent). FrictionDRBamActuator accumulates I^2 over
the physics substeps (tasks/mdp.servo_heat_state reads the per-env-step mean, and the mean signed current). Only the I^2 R part of the electrical power
heats the servo; V * I - I^2 R is mechanical (back-EMF) power.

The Dynamixel "Present Temperature" sensor reads the controller board inside the case, so the shutdown is judged on
the HOUSING node (T_h >= 70 C = XL330 Temperature Limit (31) default 70 C, Shutdown (63) default 53 = overheating bit
set; ROBOTIS e-manual XL330-M288, read 2026-11-06). The winding node is reported too.

Parameters are NOT measured on a Microduck servo. Sources / assumptions:
  - R0, kt: BAM fit (above). ALPHA_CU: copper, 0.0039 / K.
  - C_w ~ 1 J/K: ~2 g of copper + rotor iron in the ~12 mm coreless/iron-core motor (0.39-0.45 J/(g K)).
  - C_h ~ 10 J/K: the 18 g servo minus the winding (plastic case ~1.3-1.5 J/(g K), steel/brass gears 0.4-0.5).
  - R_ha ~ 35 K/W: 20 x 34 x 26 mm case, 42 cm^2, natural convection h ~ 7-10 W/(m^2 K) -> 24-34 K/W free; mounted in
    the robot frame next to other servos, part of the area is shielded -> 25-50 K/W.
  - R_wh ~ 12 K/W: small DC motors list 10-25 K/W winding-to-case.
 a nominal servo held at 1.0 A reaches 70 C in ~3 min, at
the datasheet 5 V stall current (1.47 A) in ~75 s; 0.5 A RMS is sustainable (steady state 55 C at 25 C ambient), the
nominal continuous limit is ~0.66 A RMS (1.24 W; 0.46 A / 0.65 W in the hot corner). ``XL330_HOT`` is the pessimistic corner used for "time to shutdown" worst cases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import math

import torch

R0_OHM = 2.8113923539223227        # bam/params/xl330/m6.json "R"
KT_NM_PER_A = 0.36601349688984386  # bam/params/xl330/m6.json "kt"
ALPHA_CU = 0.0039                  # 1/K, copper resistance tempco
T_REF = 25.0                       # C, temperature of the R0 fit (room)
T_LIMIT = 70.0                     # C, XL330 Temperature Limit (31) default -> overheating shutdown


@dataclass(frozen=True)
class ThermalParams:
    c_w: float = 1.0        # J/K   winding
    c_h: float = 10.0       # J/K   housing + PCB (sensor node)
    r_wh: float = 12.0      # K/W   winding -> housing
    r_ha: float = 35.0      # K/W   housing -> ambient
    t_amb: float = 25.0     # C
    p_idle: float = 0.05    # W     controller / LED standby into the housing node
    r_scale: float = 1.0    # x R0  (driver H-bridge + wiring losses folded in, unit spread)


XL330_NOMINAL = ThermalParams()
# Pessimistic corner: warm room / enclosed robot, poorly ventilated mount, smaller heat capacities, +10 % resistance.
XL330_HOT = ThermalParams(c_w=0.7, c_h=7.0, r_wh=18.0, r_ha=50.0, t_amb=35.0, p_idle=0.05, r_scale=1.1)
# Ranges for randomization / sensitivity (lo, hi).
XL330_RANGES = dict(c_w=(0.6, 1.5), c_h=(7.0, 14.0), r_wh=(8.0, 18.0), r_ha=(25.0, 50.0), t_amb=(20.0, 35.0),
                    r_scale=(0.95, 1.15))


def copper_power(i2: torch.Tensor, t_w: torch.Tensor | float | None, p: ThermalParams = XL330_NOMINAL) -> torch.Tensor:
    """I^2 R(T) loss (W). ``t_w`` None -> R at T_REF."""
    r = R0_OHM * p.r_scale
    if t_w is not None:
        r = r * (1.0 + ALPHA_CU * (torch.as_tensor(t_w) - T_REF))
    return i2 * r


def continuous_power_limit(p: ThermalParams = XL330_NOMINAL, t_limit: float = T_LIMIT) -> float:
    """Steady-state copper loss (W) that holds the housing node exactly at t_limit (idle power included)."""
    return (t_limit - p.t_amb) / p.r_ha - p.p_idle


class ServoThermal:
    """Batched two-node model, state (N, J) on any device. Explicit Euler (tau_w = C_w R_wh ~ 12 s >> dt)."""

    def __init__(self, shape, device, p: ThermalParams = XL330_NOMINAL, substeps: int = 1):
        self.p = p
        self.t_w = torch.full(shape, p.t_amb, device=device)
        self.t_h = torch.full(shape, p.t_amb, device=device)
        self.peak_h = self.t_h.clone()
        self.peak_w = self.t_w.clone()
        self.t_over = torch.full(shape, float("nan"), device=device)   # first time T_h >= T_LIMIT (s)
        self.time = 0.0
        self.substeps = substeps

    def step(self, i2: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance by dt with mean I^2 (A^2) over the step; returns the copper power (W)."""
        p = self.p
        h = dt / self.substeps
        pw = None
        for _ in range(self.substeps):
            pw = copper_power(i2, self.t_w, p)
            q_wh = (self.t_w - self.t_h) / p.r_wh
            q_ha = (self.t_h - p.t_amb) / p.r_ha
            self.t_w = self.t_w + h * (pw - q_wh) / p.c_w
            self.t_h = self.t_h + h * (q_wh - q_ha + p.p_idle) / p.c_h
        self.time += dt
        self.peak_h = torch.maximum(self.peak_h, self.t_h)
        self.peak_w = torch.maximum(self.peak_w, self.t_w)
        newly = torch.isnan(self.t_over) & (self.t_h >= T_LIMIT)
        self.t_over = torch.where(newly, torch.full_like(self.t_over, self.time), self.t_over)
        return pw


def time_to_limit_const_current(i_amp: float, p: ThermalParams = XL330_NOMINAL, dt: float = 0.05,
                                t_max: float = 3600.0) -> float:
    """Seconds until the housing node reaches T_LIMIT at a constant current from ambient (inf if never)."""
    m = ServoThermal((1,), "cpu", p)
    i2 = torch.tensor([i_amp * i_amp])
    t = 0.0
    while t < t_max:
        m.step(i2, dt)
        t += dt
        if m.t_h.item() >= T_LIMIT:
            return t
    return math.inf


def steady_state_const_current(i_amp: float, p: ThermalParams = XL330_NOMINAL) -> tuple[float, float]:
    """(T_w, T_h) steady state at a constant current (fixed point with the copper tempco), C."""
    t_w = p.t_amb
    for _ in range(200):
        pw = i_amp * i_amp * R0_OHM * p.r_scale * (1.0 + ALPHA_CU * (t_w - T_REF))
        t_h = p.t_amb + (pw + p.p_idle) * p.r_ha
        t_w_new = t_h + pw * p.r_wh
        if t_w_new > 1000.0:
            return math.inf, math.inf
        if abs(t_w_new - t_w) < 1e-6:
            break
        t_w = t_w_new
    return t_w, t_h


def with_overrides(p: ThermalParams, **kw) -> ThermalParams:
    return replace(p, **kw)
