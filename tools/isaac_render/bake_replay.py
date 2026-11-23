# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Bake the edit (a timeline .npz from edit_cine.py) into a USD replay over the skatepark (Isaac Sim python).

    <isaac-sim python> tools\\isaac_render\\bake_replay.py [--timeline T] [--out scenes\\skatepark\\replay.usda]

The stage sublayers park.usda and adds, all time-sampled at the edit's fps (no physics: what MuJoCo simulated):
    /World/Microduck/Robot   the Microduck USD; trunk pose + 14 joint angles per frame
    /World/Skateboard/<body> every board body pose per frame, drawn from the MuJoCo geoms with real materials
    /World/ShotCamera        the edit's camera (pose, focal length, focus distance, f-stop)
Then it checks the baked robot against MuJoCo: every link origin vs MuJoCo's body xpos (prints the max error).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent  # the repo root: scenes/, output/ and assets/ live there
# the robot as USD, made once with Isaac Sim's MJCF importer (see tools/isaac_render/README.md)
ROBOT_USD = Path(os.environ.get("MICRODUCK_ROBOT_USD", str(PROJECT / "assets/microduck_usd/robot_allcollisions/robot_allcollisions.usda")))
TEX = PROJECT / "assets" / "skatepark" / "tex"

parser = argparse.ArgumentParser()
parser.add_argument("--timeline", default=str(HERE / "timeline.npz"))
parser.add_argument("--park", default=str(PROJECT / "scenes" / "skatepark" / "park.usda"))
parser.add_argument("--out", default=str(PROJECT / "scenes" / "skatepark" / "replay.usda"))
parser.add_argument("--frames", help="a:b, bake only these output frames (look-dev)")
args, _ = parser.parse_known_args()

try:                                  # Isaac Sim python; plain usd-core (.usdvenv: no renderer, no GPU) also works
    from isaacsim import SimulationApp  # noqa: E402
    app = SimulationApp({"headless": True})
except ImportError:
    app = None

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

tl = np.load(args.timeline)
N = len(tl["shot"])
lo, hi = (map(int, args.frames.split(":")) if args.frames else (0, N))
FR = range(lo, min(hi, N))
fps = int(tl["fps"])

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    out.unlink()
stage = Usd.Stage.CreateNew(str(out))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.GetRootLayer().subLayerPaths.append(str(Path(args.park).resolve()).replace("\\", "/"))
stage.SetDefaultPrim(stage.OverridePrim("/World"))
stage.SetTimeCodesPerSecond(fps)
stage.SetFramesPerSecond(fps)
stage.SetStartTimeCode(FR.start)
stage.SetEndTimeCode(FR.stop - 1)
stage.GetRootLayer().customLayerData = {"shots": str(tl["shots"]), "timeline": str(Path(args.timeline).resolve())}


# ── materials ────────────────────────────────────────────────────────────────
def omnipbr(key, color=None, base=None, normal=None, orm=None, rough=0.5, metal=0.0, tint=(1, 1, 1), spec=0.5,
            bump=1.0, clearcoat=False):
    path = f"/World/Looks/Board_{key}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    i = lambda n, t, v: sh.CreateInput(n, t).Set(v)
    if color is not None:
        i("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color))
    if base:
        i("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(base))
        sh.GetInput("diffuse_texture").GetAttr().SetColorSpace("sRGB")
    if normal:
        i("normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(normal))
        sh.GetInput("normalmap_texture").GetAttr().SetColorSpace("raw")
        i("bump_factor", Sdf.ValueTypeNames.Float, float(bump))
    if orm:
        i("ORM_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(orm))
        sh.GetInput("ORM_texture").GetAttr().SetColorSpace("raw")
        i("enable_ORM_texture", Sdf.ValueTypeNames.Bool, True)
        i("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.0)
    else:
        i("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 0.0)
    i("metallic_texture_influence", Sdf.ValueTypeNames.Float, 0.0)
    i("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*tint))
    i("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rough))
    i("metallic_constant", Sdf.ValueTypeNames.Float, float(metal))
    i("specular_level", Sdf.ValueTypeNames.Float, float(spec))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def t(name, kind):
    for k in ([kind] if isinstance(kind, str) else kind):
        p = TEX / name / f"{name}_{k}.png"
        if p.exists():
            return str(p).replace("\\", "/")


BM = dict(
    wood=omnipbr("Maple", base=t("Plywood", "BaseColor"), normal=t("Plywood", "N"), orm=t("Plywood", "ORM"),
                 tint=(1.05, 0.95, 0.82)),
    grip=omnipbr("Grip", base=t("Asphalt", "BaseColor"), normal=t("Asphalt", "N"), tint=(0.16, 0.16, 0.17), rough=0.92,
                 spec=0.3, bump=1.5),
    alu=omnipbr("Truck", base=t("Aluminum_Cast", "BaseColor"), normal=t("Aluminum_Cast", "N"), metal=1.0, rough=0.35,
                tint=(0.9, 0.9, 0.92)),
    steel=omnipbr("Axle", color=(0.55, 0.56, 0.58), metal=1.0, rough=0.25),
    bushing=omnipbr("Bushing", color=(0.85, 0.2, 0.12), rough=0.45),
    wheel=omnipbr("Wheel", color=(0.12, 0.62, 0.72), rough=0.38, spec=0.6),
    hub=omnipbr("Hub", color=(0.9, 0.9, 0.88), rough=0.3),
    grip_plain=omnipbr("GripPlain", color=(0.035, 0.035, 0.038), rough=0.92, spec=0.3),
    wood_plain=omnipbr("MaplePlain", color=(0.78, 0.6, 0.38), rough=0.55),
    art=omnipbr("DeckArt", base=str(TEX / "gen" / "deck_art.png").replace("\\", "/"), rough=0.4, spec=0.6),
)


def board_material(name, rgba):
    n = name.lower()
    if "grip" in n:
        return BM["grip"]
    if "wheel" in n and "hub" in n:
        return BM["hub"]
    if "wheel" in n:
        return BM["wheel"]
    if "bushing" in n:
        return BM["bushing"]
    if "axle" in n:
        return BM["steel"]
    if any(k in n for k in ("hanger", "kingpin", "baseplate")):
        return BM["alu"]
    return BM["wood"]


# ── board ────────────────────────────────────────────────────────────────────
def box_mesh(path, half, tile=0.05, art_bottom=False):
    hx, hy, hz = half
    c = [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    k = lambda sx, sy, sz: (sx > 0) * 4 + (sy > 0) * 2 + (sz > 0)
    faces = [[k(-1, -1, 1), k(1, -1, 1), k(1, 1, 1), k(-1, 1, 1)], [k(-1, -1, -1), k(-1, 1, -1), k(1, 1, -1), k(1, -1, -1)],
             [k(-1, -1, -1), k(1, -1, -1), k(1, -1, 1), k(-1, -1, 1)], [k(-1, 1, -1), k(-1, 1, 1), k(1, 1, 1), k(1, 1, -1)],
             [k(-1, -1, -1), k(-1, -1, 1), k(-1, 1, 1), k(-1, 1, -1)], [k(1, -1, -1), k(1, 1, -1), k(1, 1, 1), k(1, -1, 1)]]
    normals = [(0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
    pts, idx, st, nrm = [], [], [], []
    for fi, f in enumerate(faces):
        n = normals[fi]
        a = int(np.argmax(np.abs(n)))
        ax = [i for i in range(3) if i != a]
        for v in f:
            p = c[v]
            pts.append(p)
            idx.append(len(pts) - 1)
            st.append((p[ax[0]] / tile, p[ax[1]] / tile))
            nrm.append(n)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        Vt.Vec2fArray([Gf.Vec2f(*s) for s in st]))
    m.CreateExtentAttr([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, hy, hz)])
    return m


def lathe_mesh(path, radius, half, fillet, n=48):
    """Smooth cylinder about local z (MuJoCo cylinder frame) with rounded edges: a lathe of the profile
    (0,-h) -> (R-f,-h) -> arc -> (R,-h+f) -> (R,h-f) -> arc -> (R-f,h) -> (0,h). Hard edges where the profile turns
    by more than 40 deg, smooth elsewhere (faceVarying normals)."""
    R, h, f = radius, half, min(fillet, 0.45 * half, 0.45 * radius)
    prof = [(0.0, -h)]
    if f > 1e-6:
        prof.append((R - f, -h))
        for k in range(1, 7):
            a = -math.pi / 2 + (math.pi / 2) * k / 6
            prof.append((R - f + f * math.cos(a), -h + f + f * math.sin(a)))
        prof.append((R, h - f))
        for k in range(1, 7):
            a = (math.pi / 2) * k / 6
            prof.append((R - f + f * math.cos(a), h - f + f * math.sin(a)))
    else:
        prof += [(R, -h), (R, h)]
    prof.append((0.0, h))
    prof = np.array(prof)
    seg_n = []
    for j in range(len(prof) - 1):
        dr, dz = prof[j + 1] - prof[j]
        L = math.hypot(dr, dz)
        seg_n.append((dz / L, -dr / L))                      # outward normal (r, z) of the profile segment
    seg_n = np.array(seg_n)

    def vn(j, s):   # normal at profile vertex j for segment s (smooth with the neighbour if the turn is small)
        o = s - 1 if j == s else s + 1
        if 0 <= o < len(seg_n) and seg_n[o] @ seg_n[s] > math.cos(math.radians(40)):
            v = seg_n[o] + seg_n[s]
            return v / np.linalg.norm(v)
        return seg_n[s]
    pts, idx, nrm, st = [], [], [], []
    for s in range(len(prof) - 1):
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            for (j, a) in ((s, a0), (s, a1), (s + 1, a1), (s + 1, a0)):
                r, z = prof[j]
                nr, nz = vn(j, s)
                pts.append((r * math.cos(a), r * math.sin(a), z))
                nrm.append((nr * math.cos(a), nr * math.sin(a), nz))
                st.append((a / (2 * math.pi), z / (2 * h) + 0.5))
                idx.append(len(pts) - 1)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr([4] * (len(idx) // 4))
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in nrm]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        Vt.Vec2fArray([Gf.Vec2f(*v) for v in st]))
    m.CreateExtentAttr([Gf.Vec3f(-R, -R, -h), Gf.Vec3f(R, R, h)])
    return m


def _grid_mesh(path, P, N, ST, double=True):
    """Quad mesh from a (nu, nv) grid of points P, smooth normals N, uvs ST (all faceVarying)."""
    nu, nv = P.shape[:2]
    pts, nrm, st, idx = [], [], [], []
    for i in range(nu - 1):
        for j in range(nv - 1):
            for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                pts.append(Gf.Vec3f(*map(float, P[a, b])))
                nrm.append(Gf.Vec3f(*map(float, N[a, b])))
                st.append(Gf.Vec2f(*map(float, ST[a, b])))
                idx.append(len(pts) - 1)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr([4] * (len(idx) // 4))
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr(Vt.Vec3fArray(nrm))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(double)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        Vt.Vec2fArray(st))
    lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    return m


def deck_meshes(root):
    """One continuous deck in the skateboard body frame, following the MuJoCo deck geoms' mid-surface (flat 9.4 cm
    platform, 37.6 deg risers to the raised ends, 18 deg kicks, rounded nose / tail), instead of the separate boxes
    and discs the MJCF uses (which render as stacked plates with steps and gaps). Wood shell + grip top + bottom art."""
    UsdGeom.Xform.Define(stage, root)
    L, th = 0.112, 0.002
    # mid-surface centreline (x >= 0, mirrored), corners rounded by smoothing
    kx = np.array([0.0, 0.044, 0.062, 0.079, 0.097, L])
    kz = np.array([0.0, 0.0, 0.0100, 0.0100, 0.0159, 0.0208])
    xs = np.linspace(-L, L, 161)
    zc = np.interp(np.abs(xs), kx, kz)
    dx = xs[1] - xs[0]
    sig = 0.003 / dx
    r = int(3 * sig)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    zc = np.convolve(np.pad(zc, r, mode="reflect", reflect_type="odd"), k, mode="valid")
    ax = np.abs(xs)
    w = np.where(ax <= 0.040, 0.055, np.where(ax >= 0.062, 0.035,
                 0.035 + 0.020 * 0.5 * (1 + np.cos(math.pi * (ax - 0.040) / 0.022))))
    tip = ax > 0.097
    w[tip] = 0.020 + np.sqrt(np.maximum(0.015 ** 2 - (ax[tip] - 0.097) ** 2, 0.0))
    w = np.maximum(w, 0.0205)
    dz = np.gradient(zc, dx)
    dw = np.gradient(w, dx) * np.sign(xs + 1e-12)
    nv = 13
    v = np.linspace(-1, 1, nv)
    Y = w[:, None] * v[None]
    X = np.repeat(xs[:, None], nv, 1)
    tn = np.stack([-dz, np.zeros_like(dz), np.ones_like(dz)], 1)
    tn /= np.linalg.norm(tn, axis=1, keepdims=True)
    up = np.repeat(tn[:, None], nv, 1)
    def surf(off, shrink=1.0):
        return np.stack([X, Y * shrink, np.broadcast_to(zc[:, None] + off, X.shape)], 2)
    st_top = np.stack([X / 0.06, Y / 0.06], 2)
    # wood shell: top, bottom, two sides, two end caps
    top, bot = surf(th), surf(-th)
    wood = [(_grid_mesh(f"{root}/DeckTop", top, up, st_top), BM["wood"]),
            (_grid_mesh(f"{root}/DeckBottom", bot[:, ::-1], -up[:, ::-1], st_top[:, ::-1]), BM["wood"])]
    wp = np.gradient(w, dx)                       # dw/dx (true derivative)
    for sgn, nm in ((1, "DeckSideL"), (-1, "DeckSideR")):
        jj = nv - 1 if sgn > 0 else 0
        P = np.stack([top[:, jj], bot[:, jj]], 1)                      # (nu, 2, 3)
        n = np.stack([-wp, np.full_like(wp, float(sgn)), np.zeros_like(wp)], 1)   # outward of y = sgn * w(x)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        Nn = np.repeat(n[:, None], 2, 1)
        STs = np.stack([P[..., 0] / 0.06, P[..., 2] / 0.06], 2)
        wood.append((_grid_mesh(f"{root}/{nm}", P, Nn, STs), BM["wood"]))
    for e, nm, nx in ((0, "DeckCapR", -1.0), (-1, "DeckCapF", 1.0)):
        P = np.stack([top[e], bot[e]], 0)                              # (2, nv, 3)
        Nn = np.broadcast_to(np.array([nx, 0.0, 0.0]), P.shape).copy()
        STs = np.stack([P[..., 1] / 0.06, P[..., 2] / 0.06], 2)
        wood.append((_grid_mesh(f"{root}/{nm}", P, Nn, STs), BM["wood"]))
    # grip tape: inset 1.2 mm from the edge, 0.3 mm above the wood
    grip = surf(th + 0.0003, shrink=1.0)
    grip[..., 1] = (w[:, None] - 0.0012) * v[None]
    gx = (np.abs(xs) <= L - 0.0012)
    out = wood + [(_grid_mesh(f"{root}/Grip", grip[gx], up[gx], st_top[gx]), BM["grip"])]
    # bottom graphic: whole art image over the bottom (planar u along x, v across the widest point)
    art = surf(-th - 0.0003)
    art[..., 1] = (w[:, None] - 0.001) * v[None]
    st_art = np.stack([(X + L) / (2 * L), (art[..., 1] + 0.055) / 0.11], 2)
    out.append((_grid_mesh(f"{root}/DeckArt", art[:, ::-1], -up[:, ::-1], st_art[:, ::-1]), BM["art"]))
    for m, mat in out:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def deck_meshes_from_geoms(root, idx):
    """One continuous deck lofted around the MuJoCo deck geoms of the sequence board (flat deck + wide tail pad +
    tapers + kicked nose / tail with round corners): the top view outline w(x), top z_top(x) and bottom z_bot(x) are
    sampled from every deck box / corner disc in the skateboard body frame, then lofted like deck_meshes (wood shell,
    grip top inset 1.2 mm, bottom art)."""
    UsdGeom.Xform.Define(stage, root)
    gt, size, pos, quat = (tl[f"board_geom_{k}"] for k in ("type", "size", "pos", "quat"))
    top, bot = [], []
    for i in idx:
        w_, x_, y_, z_ = map(float, quat[i])
        R = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(w_, x_, y_, z_))).GetTranspose())   # local -> body columns
        c = np.array(pos[i], float)
        if str(gt[i]) == "box":
            hx, hy, hz = map(float, size[i])
            u, v = np.meshgrid(np.linspace(-hx, hx, 60), np.linspace(-hy, hy, 30))
            for zz, dst in ((hz, top), (-hz, bot)):
                P = np.stack([u.ravel(), v.ravel(), np.full(u.size, zz)], 1)
                dst.append(P @ R.T + c)
        elif str(gt[i]) == "cylinder":
            r, hl = float(size[i][0]), float(size[i][1])
            rr, aa = np.meshgrid(np.linspace(0, r, 20), np.linspace(0, 2 * math.pi, 72))
            for zz, dst in ((hl, top), (-hl, bot)):
                P = np.stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel(), np.full(rr.size, zz)], 1)
                dst.append(P @ R.T + c)
    top, bot = np.concatenate(top), np.concatenate(bot)
    x_lo, x_hi = float(min(top[:, 0].min(), bot[:, 0].min())), float(max(top[:, 0].max(), bot[:, 0].max()))
    xs = np.linspace(x_lo + 2e-4, x_hi - 2e-4, 181)
    hb = (xs[1] - xs[0]) * 0.75
    w, zt, zb = np.full(len(xs), np.nan), np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    for k, x in enumerate(xs):
        a = top[np.abs(top[:, 0] - x) <= hb]
        b = bot[np.abs(bot[:, 0] - x) <= hb]
        if len(a):
            w[k], zt[k] = np.abs(a[:, 1]).max(), a[:, 2].max()
        if len(b):
            zb[k] = b[:, 2].min()
    ok = ~np.isnan(w) & ~np.isnan(zb)
    w, zt, zb = (np.interp(xs, xs[ok], v[ok]) for v in (w, zt, zb))

    def sm(v, sig_m):
        sig = sig_m / (xs[1] - xs[0])
        r = int(3 * sig)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        k /= k.sum()
        return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")
    w = np.maximum(sm(w, 0.0025), 0.004)
    zt, zb = sm(zt, 0.003), sm(zb, 0.006)
    zb = np.minimum(zb, zt - 0.003)
    # round the ends in top view: shrink the width over the last 3 mm (the sampled tips are flat)
    dx = xs[1] - xs[0]
    nv = 15
    v = np.linspace(-1, 1, nv)
    X = np.repeat(xs[:, None], nv, 1)
    Y = w[:, None] * v[None]
    dzt = np.gradient(zt, dx)
    up1 = np.stack([-dzt, np.zeros_like(dzt), np.ones_like(dzt)], 1)
    up1 /= np.linalg.norm(up1, axis=1, keepdims=True)
    up = np.repeat(up1[:, None], nv, 1)
    T = np.stack([X, Y, np.broadcast_to(zt[:, None], X.shape)], 2)
    B = np.stack([X, Y, np.broadcast_to(zb[:, None], X.shape)], 2)
    st_top = np.stack([X / 0.06, Y / 0.06], 2)
    parts = [(_grid_mesh(f"{root}/DeckTop", T, up, st_top), BM["wood"]),
             (_grid_mesh(f"{root}/DeckBottom", B[:, ::-1], -up[:, ::-1], st_top[:, ::-1]), BM["wood"])]
    wp = np.gradient(w, dx)
    for sgn, nm in ((1, "DeckSideL"), (-1, "DeckSideR")):
        jj = nv - 1 if sgn > 0 else 0
        P = np.stack([T[:, jj], B[:, jj]], 1)
        n = np.stack([-wp, np.full_like(wp, float(sgn)), np.zeros_like(wp)], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        STs = np.stack([P[..., 0] / 0.06, P[..., 2] / 0.06], 2)
        parts.append((_grid_mesh(f"{root}/{nm}", P, np.repeat(n[:, None], 2, 1), STs), BM["wood"]))
    for e, nm, nx in ((0, "DeckCapR", -1.0), (-1, "DeckCapF", 1.0)):
        P = np.stack([T[e], B[e]], 0)
        Nn = np.broadcast_to(np.array([nx, 0.0, 0.0]), P.shape).copy()
        parts.append((_grid_mesh(f"{root}/{nm}", P, Nn, np.stack([P[..., 1] / 0.06, P[..., 2] / 0.06], 2)), BM["wood"]))
    G = T.copy()
    G[..., 1] = np.maximum(w[:, None] - 0.0012, 0.002) * v[None]
    G[..., 2] += 0.0003
    gx = (xs >= x_lo + 0.0015) & (xs <= x_hi - 0.0015)
    parts.append((_grid_mesh(f"{root}/Grip", G[gx], up[gx], st_top[gx]), BM["grip"]))
    A = B.copy()
    A[..., 1] = np.maximum(w[:, None] - 0.001, 0.002) * v[None]
    A[..., 2] -= 0.0003
    wm = float(w.max())
    st_art = np.stack([(X - x_lo) / (x_hi - x_lo), (A[..., 1] + wm) / (2 * wm)], 2)
    parts.append((_grid_mesh(f"{root}/DeckArt", A[:, ::-1], -up[:, ::-1], st_art[:, ::-1]), BM["art"]))
    for m, mat in parts:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    print(f"BAKE deck loft from {len(idx)} geoms: x {x_lo:.4f}..{x_hi:.4f}, width {2 * w.min():.3f}-{2 * wm:.3f}, "
          f"top z {zt.min():.4f}..{zt.max():.4f}, thickness {(zt - zb).min() * 1000:.1f}-{(zt - zb).max() * 1000:.1f} mm",
          flush=True)


def add_board():
    names = [str(n) for n in tl["board_body_names"]]
    gb, gn, gt = (tl[f"board_geom_{k}"] for k in ("body", "name", "type"))
    size, pos, quat, rgba = (tl[f"board_geom_{k}"] for k in ("size", "pos", "quat", "rgba"))
    parts = []
    for b in names:
        xf = UsdGeom.Xform.Define(stage, f"/World/Skateboard/{b}")
        parts.append((xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble),
                      xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)))
        seq = any(str(n).endswith("deck_tailpad_collision") for n in gn)     # the sequence task's board
        if b == "skateboard":
            if seq:
                deck_meshes_from_geoms(f"/World/Skateboard/{b}/Deck", [
                    i for i in range(len(gb)) if str(gb[i]) == b and str(gn[i]).startswith("deck_")
                    and "bolt" not in str(gn[i]) and "stripe" not in str(gn[i]) and "grip" not in str(gn[i])])
            else:
                deck_meshes(f"/World/Skateboard/{b}/Deck")
        for i in [i for i in range(len(gb)) if str(gb[i]) == b]:
            nm_ = str(gn[i]).split("/")[-1]
            if b == "skateboard" and nm_.startswith("deck_") and not (seq and nm_.startswith("deck_bolt")):
                continue                 # replaced by the continuous deck (the sequence board keeps its bolts)
            nm = "".join(ch if ch.isalnum() else "_" for ch in str(gn[i])) or f"g{i}"
            p = f"/World/Skateboard/{b}/{nm}"
            ty = str(gt[i])
            if ty == "box":
                prim = box_mesh(p, tuple(map(float, size[i])), tile=0.06)
            elif ty == "cylinder":
                r, hl = float(size[i][0]), float(size[i][1])
                nl = str(gn[i]).lower()
                fil = 0.35 * hl if ("wheel" in nl and "hub" not in nl) else 0.15 * min(r, hl)
                prim = lathe_mesh(p, r, hl, fil, n=56 if r > 0.003 else 16)
            elif ty == "capsule":
                prim = UsdGeom.Capsule.Define(stage, p)
                prim.CreateAxisAttr("Z")
                prim.CreateRadiusAttr(float(size[i][0]))
                prim.CreateHeightAttr(float(2 * size[i][1]))
            else:
                print(f"BAKE skip geom {nm} ({ty})")
                continue
            x = UsdGeom.Xformable(prim)
            x.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos[i])))
            w, qx, qy, qz = map(float, quat[i])
            x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, qx, qy, qz))
            mat = board_material(str(gn[i]), rgba[i])
            if ty == "capsule":          # no UVs on capsules: textured grip smears radially
                mat = BM["grip_plain"] if mat is BM["grip"] else BM["wood_plain"]
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
            if str(gn[i]).endswith("deck_collision"):   # deck graphic on the bottom
                art = box_mesh(f"/World/Skateboard/{b}/deck_art", (float(size[i][0]) - 0.001, float(size[i][1]) - 0.001,
                                                                     0.0002), tile=1.0)
                ax = UsdGeom.Xformable(art)
                ax.AddTranslateOp().Set(Gf.Vec3d(float(pos[i][0]), float(pos[i][1]), float(pos[i][2] - size[i][2] - 0.0003)))
                # UV the bottom face over the whole art image
                st = []
                for fi in range(6):
                    for (u, v) in ((0, 0), (0, 1), (1, 1), (1, 0)):
                        st.append((u, v))
                UsdGeom.PrimvarsAPI(art).GetPrimvar("st").Set(Vt.Vec2fArray([Gf.Vec2f(*s) for s in st]))
                UsdShade.MaterialBindingAPI.Apply(art.GetPrim()).Bind(BM["art"])
    return parts


# ── robot ────────────────────────────────────────────────────────────────────
robot = stage.DefinePrim("/World/Microduck/Robot", "Xform")
robot.GetReferences().AddReference(str(ROBOT_USD).replace("\\", "/"))
robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
UsdGeom.Xform.Define(stage, "/World/Microduck")

targets = robot.GetRelationship("isaac:physics:robotJoints").GetTargets()
links = {p.name: stage.GetPrimAtPath(p.GetParentPath()) for p in targets}
names = [str(n) for n in tl["joint_names"]]
assert sorted(links) == sorted(names), (sorted(links), names)
trunk = stage.GetPrimAtPath(robot.GetPath().AppendPath("Geometry/trunk_base"))
tr_t, tr_o = trunk.GetAttribute("xformOp:translate"), trunk.GetAttribute("xformOp:orient")
TQ = type(tr_o.Get())
anim = []
for j, n in enumerate(names):
    o = links[n].GetAttribute("xformOp:orient")
    anim.append((o, Gf.Quatd(o.Get()), type(o.Get()), j))

# robot look: 3D-printed plastic is a little rougher than the converter's 0.5
for prim in stage.Traverse():
    if prim.GetTypeName() == "Shader" and str(prim.GetPath()).startswith("/World/Microduck"):
        r = prim.GetAttribute("inputs:roughness")
        if r and r.IsValid():
            r.Set(0.62)

board_parts = add_board()

# ── camera ───────────────────────────────────────────────────────────────────
cam = UsdGeom.Camera.Define(stage, "/World/ShotCamera")
ha, va = (float(v) for v in tl["aperture"]) if "aperture" in tl.files else (36.0, 20.25)   # 9x16: 20.25 x 36
cam.CreateHorizontalApertureAttr(ha)
cam.CreateVerticalApertureAttr(va)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 2000.0))
cam_xf = cam.AddTransformOp()
fl, fd, fs = cam.CreateFocalLengthAttr(), cam.CreateFocusDistanceAttr(), cam.CreateFStopAttr()

for k in FR:
    tr_t.Set(Gf.Vec3d(*tl["root_pos"][k].tolist()), k)
    w, x, y, z = tl["root_quat"][k].tolist()
    tr_o.Set(TQ(Gf.Quatd(w, x, y, z)), k)
    for o, rest, Q, j in anim:
        h = float(tl["joint_pos"][k, j]) / 2
        o.Set(Q(rest * Gf.Quatd(math.cos(h), 0, 0, math.sin(h))), k)
    for b, (bt, bo) in enumerate(board_parts):
        bt.Set(Gf.Vec3d(*tl["board_pos"][k, b].tolist()), k)
        w, x, y, z = tl["board_quat"][k, b].tolist()
        bo.Set(Gf.Quatd(w, x, y, z), k)
    eye, tgt = Gf.Vec3d(*tl["eye"][k].tolist()), Gf.Vec3d(*tl["target"][k].tolist())
    cam_xf.Set(Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse(), k)
    fl.Set(float(tl["focal"][k]), k)
    fd.Set(float((eye - tgt).GetLength()), k)
    # USD lens units are tenths of a scene unit (mm in cm scenes): in this meter stage focalLength 50 reads as a 5 m
    # lens, so the physical aperture (focal / fStop) is 100x too big unless fStop is scaled by the same 100
    fs.Set(float(tl["fstop"][k]) * 100.0, k)

stage.GetRootLayer().Save()

# ── check: baked link origins vs MuJoCo body positions ─────────────────────
body_names = [str(n) for n in tl["robot_body_names"]]
usd_links = {p.GetName(): p for p in [trunk] + [links[n] for n in names]}
for p in list(usd_links.values()):
    for c in Usd.PrimRange(p):
        if c.HasAPI(UsdGeom.Xformable) or c.IsA(UsdGeom.Xformable):
            pass
errs = []
pick = list(FR)[:: max(1, len(FR) // 40)]
missing = set()
for k in pick:
    cache = UsdGeom.XformCache(k)
    for bi, bn in enumerate(body_names):
        prim = stage.GetPrimAtPath(trunk.GetPath()) if bn == "trunk_base" else None
        if prim is None:
            cands = [lp for lp in usd_links.values() if lp.GetName() in (bn, bn + "_1")]
            if not cands:
                missing.add(bn)
                continue
            prim = cands[0]
        p = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        errs.append(float(np.linalg.norm(np.array(p) - tl["robot_body_pos"][k, bi])))
print(f"BAKE check: {len(pick)} frames, max link error {max(errs) * 1000:.2f} mm, mean {np.mean(errs) * 1000:.3f} mm"
      + (f", unmatched bodies {sorted(missing)}" if missing else ""), flush=True)
print(f"BAKE wrote {out}: frames {FR.start}-{FR.stop - 1} ({len(FR) / fps:.1f} s at {fps} fps)", flush=True)
if app is not None:
    app.close()
