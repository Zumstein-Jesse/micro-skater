# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Mini half-pipe geometry for MuJoCo / MuJoCo Warp (2026-11-19, sequence landscape study).

Profile (xz, constant along y): flat bottom |x| <= L/2 at z = 0, then on each side a circular transition of radius R
rising to height h (angle phi_max = acos(1 - h / R)), then a short flat platform ("deck") at z = h. The floor plane
around it is at z = 0, so the half-pipe is open at both y ends and the robot can roll in from the side.

Three builders, all adding STATIC geoms to a body of an MjSpec (the worldbody, or mjlab's "terrain" body):
  boxes   : one thin box per facet (chord of dphi), overlapped under the neighbour. Concave joints hide every box edge
            inside the next box, so a wheel never meets an exposed edge on the transition. Cylinder-box = CONVEX (CCD)
            in MuJoCo Warp, sphere/capsule-box = PRIMITIVE (analytic).
  hfield  : one heightfield (nrow x ncol), dx resolution along x. Every pair with an hfield = CONVEX (prism CCD).
  mesh    : one convex mesh prism per facet (a concave transition must be decomposed: MuJoCo collides the convex hull).

`HalfPipeTerrainCfg` wraps the box builder as an mjlab SubTerrainCfg (same pattern as microduck_rl
tasks/slope_terrain.py FlatRampTerrainCfg): use a 1x1 TerrainGeneratorCfg so every env spawns on the same tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


def profile(x: np.ndarray, h: float, R: float, L: float, deck: float = 0.15) -> np.ndarray:
    """Surface height z(x) of the half-pipe (exact circle arcs)."""
    phi_max = math.acos(1.0 - h / R)
    x_end = L / 2 + R * math.sin(phi_max)
    ax = np.abs(x)
    z = np.zeros_like(ax)
    m = (ax > L / 2) & (ax <= x_end)
    u = ax[m] - L / 2
    z[m] = R - np.sqrt(np.maximum(R * R - u * u, 0.0))
    z[(ax > x_end) & (ax <= x_end + deck)] = h
    return z


def _facets(h: float, R: float, L: float, dphi_deg: float):
    """Chords (x0, z0, x1, z1) of the +x transition."""
    phi_max = math.acos(1.0 - h / R)
    n = max(1, int(math.ceil(math.degrees(phi_max) / dphi_deg)))
    phis = np.linspace(0.0, phi_max, n + 1)
    xs = L / 2 + R * np.sin(phis)
    zs = R * (1.0 - np.cos(phis))
    return [(xs[i], zs[i], xs[i + 1], zs[i + 1]) for i in range(n)], phi_max


def add_halfpipe_boxes(body, h=0.08, R=0.30, L=0.60, W=1.2, dphi_deg=2.0, t=0.03, overlap=0.004, deck=0.15,
                       with_bottom=False, name="hp", rgba=(0.55, 0.6, 0.7, 1.0)):
    """Add the half-pipe as boxes. Returns the list of geoms. with_bottom=False: the flat bottom is the floor plane."""
    geoms = []
    facets, phi_max = _facets(h, R, L, dphi_deg)
    k = 0
    for side in (+1, -1):
        for (x0, z0, x1, z1) in facets:
            ang = math.atan2(z1 - z0, x1 - x0)
            half = 0.5 * math.hypot(x1 - x0, z1 - z0) + overlap
            cx, cz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
            # box centre = chord midpoint moved t/2 along the inward (downward) surface normal
            nx, nz = -math.sin(ang), math.cos(ang)
            g = body.add_geom()
            g.name = f"{name}_f{k}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = (half, W / 2, t / 2)
            g.pos = (side * (cx - nx * t / 2), 0.0, cz - nz * t / 2)
            th = side * ang                         # +x side: +x end up = rotation about +y by -ang
            g.quat = (math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0)
            g.rgba = rgba
            geoms.append(g)
            k += 1
        # platform at the top (its inner edge is the coping: a CONVEX edge, keep riders below it)
        x_end = L / 2 + R * math.sin(phi_max)
        g = body.add_geom()
        g.name = f"{name}_deck{'p' if side > 0 else 'm'}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = (deck / 2, W / 2, (h + 0.01) / 2)
        g.pos = (side * (x_end + deck / 2), 0.0, (h - 0.01) / 2)
        g.rgba = rgba
        geoms.append(g)
    if with_bottom:
        g = body.add_geom()
        g.name = f"{name}_bottom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = (L / 2 + overlap, W / 2, t / 2)
        g.pos = (0.0, 0.0, -t / 2)
        g.rgba = rgba
        geoms.append(g)
    return geoms


def add_halfpipe_hfield(spec, body, h=0.08, R=0.30, L=0.60, W=1.2, dx=0.005, deck=0.15, nrow=3, base=0.05,
                        name="hp_hf"):
    """One heightfield covering the whole half-pipe (flat bottom at z=0, so the floor plane must sit at z<0 or be
    removed under it). Returns the geom."""
    phi_max = math.acos(1.0 - h / R)
    half_x = L / 2 + R * math.sin(phi_max) + deck
    ncol = int(round(2 * half_x / dx)) + 1
    x = np.linspace(-half_x, half_x, ncol)
    z = profile(x, h, R, L, deck)
    zmax = max(h, 1e-3)
    data = np.tile(z / zmax, (nrow, 1))
    hf = spec.add_hfield()
    hf.name = name
    hf.size = (half_x, W / 2, zmax, base)
    hf.nrow, hf.ncol = nrow, ncol
    hf.userdata = data.ravel().tolist()
    g = body.add_geom()
    g.name = name + "_geom"
    g.type = mujoco.mjtGeom.mjGEOM_HFIELD
    g.hfieldname = name
    g.pos = (0.0, 0.0, 0.0)
    g.rgba = (0.6, 0.55, 0.7, 1.0)
    return g


def add_halfpipe_meshes(spec, body, h=0.08, R=0.30, L=0.60, W=1.2, dphi_deg=2.0, t=0.03, overlap=0.004, deck=0.15,
                        name="hp_m"):
    """One convex mesh prism per facet (same shape as the boxes; tests cylinder-MESH)."""
    geoms = []
    facets, phi_max = _facets(h, R, L, dphi_deg)
    k = 0
    for side in (+1, -1):
        for (x0, z0, x1, z1) in facets:
            ang = math.atan2(z1 - z0, x1 - x0)
            ux, uz = math.cos(ang), math.sin(ang)
            nx, nz = -math.sin(ang), math.cos(ang)
            a = (x0 - ux * overlap, z0 - uz * overlap)
            b = (x1 + ux * overlap, z1 + uz * overlap)
            pts2 = [a, b, (b[0] - nx * t, b[1] - nz * t), (a[0] - nx * t, a[1] - nz * t)]
            verts = []
            for (px, pz) in pts2:
                for y in (-W / 2, W / 2):
                    verts += [side * px, y, pz]
            m = spec.add_mesh()
            m.name = f"{name}_mesh{k}"
            m.uservert = verts
            g = body.add_geom()
            g.name = f"{name}_f{k}"
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = m.name
            g.rgba = (0.7, 0.6, 0.5, 1.0)
            geoms.append(g)
            k += 1
        x_end = L / 2 + R * math.sin(phi_max)
        g = body.add_geom()
        g.name = f"{name}_deck{'p' if side > 0 else 'm'}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = (deck / 2, W / 2, (h + 0.01) / 2)
        g.pos = (side * (x_end + deck / 2), 0.0, (h - 0.01) / 2)
        geoms.append(g)
    return geoms


# ── mjlab sub-terrain (box builder). Import lazily so the CPU tests don't need mjlab. ──
try:
    from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

    @dataclass(kw_only=True)
    class HalfPipeTerrainCfg(SubTerrainCfg):
        """Floor slab + box half-pipe centred in the tile; origin = centre of the flat bottom.
        difficulty could scale h (keep one tile: every world holds ALL tiles' geoms)."""
        h: float = 0.08
        R: float = 0.30
        L: float = 0.60
        W: float = 1.2
        dphi_deg: float = 4.0          # 3-4 deg facets (chord ~2 cm): finer facets LOSE energy (warp_halfpipe_test)
        plane_floor: bool = True       # PLANE floor = the analytic cylinder/mesh-plane pairs the flat tasks train on

        def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
            body = spec.body("terrain")
            if self.plane_floor:
                floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(self.size[0] / 2, self.size[1] / 2, 0.1),
                                      pos=(self.size[0] / 2, self.size[1] / 2, 0.0))
            else:
                floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.size[0] / 2, self.size[1] / 2, 0.05),
                                      pos=(self.size[0] / 2, self.size[1] / 2, -0.05))
            gs = add_halfpipe_boxes(body, h=self.h, R=self.R, L=self.L, W=self.W, dphi_deg=self.dphi_deg)
            for g in gs:
                g.pos = (g.pos[0] + self.size[0] / 2, g.pos[1] + self.size[1] / 2, g.pos[2])
            geoms = [TerrainGeometry(geom=floor, color=(0.5, 0.5, 0.5, 1.0))]
            geoms += [TerrainGeometry(geom=g, color=(0.55, 0.6, 0.7, 1.0)) for g in gs]
            return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]), geometries=geoms)
except Exception:  # pragma: no cover
    HalfPipeTerrainCfg = None
