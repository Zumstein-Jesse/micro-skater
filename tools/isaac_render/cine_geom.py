# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Small geometry helpers shared by the edit step (edit_cine.py): quaternion sign continuity, interpolation,
smoothing, and the per-frame "is the robot in frame and not hidden by the park surface" check.

``ground_z(x, y)`` is the rendered park surface; edit_cine.py replaces it with the trimmed half-pipe profile.
"""
from __future__ import annotations

import numpy as np


def ground_z(x, y):
    """Flat ground (replaced by the caller with the real rendered surface)."""
    return np.zeros_like(np.atleast_1d(np.asarray(x, float)))


def fix_quat_signs(q):
    """q (N, ..., 4): flip signs so consecutive quaternions have a positive dot product."""
    q = q.copy()
    for i in range(1, len(q)):
        d = np.sum(q[i] * q[i - 1], axis=-1)
        q[i][d < 0] *= -1
    return q


def sample(arr, f):
    """Linear interpolation of arr (N, ...) at fractional indices f."""
    i0 = np.clip(np.floor(f).astype(int), 0, len(arr) - 1)
    i1 = np.clip(i0 + 1, 0, len(arr) - 1)
    w = (f - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[i0] * (1 - w) + arr[i1] * w


def nquat(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def smooth(x, sigma_steps):
    """Gaussian smoothing along axis 0 (edge padded)."""
    if sigma_steps <= 0:
        return x.copy()
    r = int(3 * sigma_steps)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_steps) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.zeros_like(x)
    for j in range(2 * r + 1):
        out += k[j] * pad[j:j + len(x)]
    return out


def look_at(eye, target):
    return np.concatenate([eye, target])


def visibility(tl, margin=0.06):
    """Per frame: is the robot (trunk, head, both feet) inside the frame (with margin) and not hidden by the park
    surface (line of sight sampled against ground_z)? Returns (in_frame, unoccluded, screen xy of the trunk)."""
    names = [str(n) for n in tl["robot_body_names"]]
    pts_idx = [names.index(n) for n in ("trunk_base", "jaw_soft", "ankle_left", "ankle_right") if n in names]
    N = len(tl["shot"])
    inf = np.zeros(N, bool)
    vis = np.zeros(N, bool)
    scr = np.zeros((N, 2))
    for k in range(N):
        eye, tgt = tl["eye"][k], tl["target"][k]
        fwd = tgt - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        tan_h = tl["aperture"][0] / 2 / tl["focal"][k]      # half aperture / focal
        tan_v = tl["aperture"][1] / 2 / tl["focal"][k]
        ok_f, ok_v = True, True
        for j, bi in enumerate(pts_idx):
            p = tl["robot_body_pos"][k, bi]
            d = p - eye
            z = d @ fwd
            if z <= 0.02:
                ok_f = False
                continue
            sx, sy = (d @ right) / z / tan_h, (d @ up) / z / tan_v
            if j == 0:
                scr[k] = (sx, sy)
            if abs(sx) > 1 - margin or abs(sy) > 1 - margin:
                ok_f = False
            s = np.linspace(0.05, 0.95, 40)
            seg = eye[None] + s[:, None] * d[None]
            if np.any(seg[:, 2] < ground_z(seg[:, 0], seg[:, 1]) - 0.003):
                ok_v = False
        inf[k], vis[k] = ok_f, ok_v
    return inf, vis, scr
