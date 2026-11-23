# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skate realism on top of microduck_rl's BAM servo model (subclass; the upstream class is imported, not copied).

Adds, all opt-in through the cfg (None = upstream behaviour):
  * ``max_current``: the firmware current limit in amps. BAM's XL330 default is 1.75 A; the robot's runtime sets
    1000 mA, so the final skate policies train with ``MICRODUCK_BAM_MAX_CURRENT=1.0``.
  * ``noload_speed_per_volt``: the datasheet no-load speed (XL330-M288-T: 103 rpm at 5 V = 2.157 rad/s per V).
    BAM uses ke = kt (0.366), i.e. ~2.73 rad/s per V, so the simulated servo was ~25 % too fast at full voltage.
  * a motor-current accumulator (``pop_i2_mean``) read by the servo heat cost (``micro_skater.servo_heat``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab_microduck.actuator.friction_dr_bam import (
    FrictionDRBamActuator as _UpstreamFrictionDRBamActuator,
    FrictionDRBamActuatorCfg as _UpstreamFrictionDRBamActuatorCfg,
)


class FrictionDRBamActuator(_UpstreamFrictionDRBamActuator):
    """Upstream FrictionDRBamActuator + current limit override, datasheet back-EMF and a current (heat) log."""

    def __init__(self, cfg, entity, target_ids, target_names) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        # The limiter lives in VoltageControlledActuator.compute_control, which BamActuator.compute calls on the GPU
        # path, so setting it on the shared BAM actuator applies it to every env.
        mc = getattr(cfg, "max_current", None)
        if mc is not None:
            prev = self._bam_model.actuator.max_current
            self._bam_model.actuator.max_current = float(mc)
            print(f"[FrictionDRBamActuator] max_current override {prev} -> {float(mc)} A", flush=True)
        # With w0 = noload_speed_per_volt the back-EMF constant becomes ke_eff = 1/w0 in both places BAM uses the
        # velocity for back-EMF (the current limiter window and kt*(V - ke*w)/R), so the torque-speed line hits zero
        # at w0*V while the stall torque kt*V/R and the kt*I_max clamp are unchanged. Implemented as an extra factor
        # ke_eff/kt on the velocity BamActuator.compute feeds those two calls (the kd_scale channel; friction and
        # Stribeck use the unscaled velocity).
        w0 = getattr(cfg, "noload_speed_per_volt", None)
        self._backemf_ratio: float | None = None
        if w0 is not None:
            kt = float(self._bam_model.kt.value)
            self._backemf_ratio = 1.0 / (float(w0) * kt)
            print(f"[FrictionDRBamActuator] datasheet no-load speed {float(w0):.3f} rad/s/V: ke_eff {1.0 / float(w0):.4f}"
                  f" (kt {kt:.4f}), back-EMF x{self._backemf_ratio:.4f}", flush=True)

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # Heat model (servo_heat.py): sum of the motor current squared over the physics substeps since the last read.
        # tau = kt * I in BAM's DC-motor equation (output referred).
        self._kt_heat = float(self._bam_model.kt.value)
        self.i2_acc: torch.Tensor | None = None
        self.i_acc: torch.Tensor | None = None
        self.i_last: torch.Tensor | None = None   # mean signed current of the last popped window (A)
        self.i2_n = 0

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        if self._backemf_ratio is None:
            tau = super().compute(cmd)
        else:
            kd = self.kd_scale
            self.kd_scale = kd * self._backemf_ratio   # back-EMF only (see __init__); restored right after
            try:
                tau = super().compute(cmd)
            finally:
                self.kd_scale = kd
        i = tau / self._kt_heat
        if self.i2_acc is None or self.i2_acc.shape != tau.shape:
            self.i2_acc = torch.zeros_like(tau)
            self.i_acc = torch.zeros_like(tau)
            self.i2_n = 0
        self.i2_acc = self.i2_acc + i * i   # out of place: no inference-tensor in-place errors
        self.i_acc = self.i_acc + i
        self.i2_n += 1
        return tau

    def pop_i2_mean(self) -> torch.Tensor | None:
        """(N, J) mean I^2 (A^2) over the substeps since the last call, then cleared. None before the first step."""
        if self.i2_acc is None or self.i2_n == 0:
            return None
        out = self.i2_acc / self.i2_n
        self.i_last = self.i_acc / self.i2_n
        self.i2_acc = torch.zeros_like(self.i2_acc)
        self.i_acc = torch.zeros_like(self.i_acc)
        self.i2_n = 0
        return out


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(_UpstreamFrictionDRBamActuatorCfg):
    """Upstream cfg + the two opt-in realism fields; builds the subclass above."""

    max_current: float | None = None  # firmware current limit [A]; None = BAM model default (XL330: 1.75 A)
    noload_speed_per_volt: float | None = None  # [rad/s/V] back-EMF ke_eff = 1/this; None = BAM ke = kt (as fitted)

    def build(self, entity, target_ids, target_names) -> FrictionDRBamActuator:
        return FrictionDRBamActuator(self, entity, target_ids, target_names)
