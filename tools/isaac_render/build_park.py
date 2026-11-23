# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Build the skatepark environment USD around the MuJoCo play park (Isaac Sim python).

    <isaac-sim python> tools\\isaac_render\\build_park.py [--out scenes\\skatepark\\park.usda]

The surfaces the robot touches are the MuJoCo park exactly: the floor plane z = 0 and the mesa (8 x 8 m, 0.2 m high)
with its four ramps, from takes/park.json (the exact ramp profile sample points, the chords MuJoCo's box segments lie
on). Everything else is dressing placed off every recorded path (checked against takes/*.kin.npz, --clearance).
The default mesa layout needs the play-sim takes (takes/park.json, takes/*.kin.npz), which are not shipped; the
chain videos use --layout pipe (chain_cine.py passes it).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent  # the repo root: scenes/, output/ and assets/ live there
ASSETS = PROJECT / "assets" / "skatepark"
TEX = ASSETS / "tex"
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/NVIDIA/Assets"

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=str(PROJECT / "scenes" / "skatepark" / "park.usda"))
parser.add_argument("--park", default=str(HERE / "takes" / "park.json"))
parser.add_argument("--clearance", type=float, default=0.6, help="m, dressing kept this far from any recorded path")
parser.add_argument("--no-remote", action="store_true", help="skip NVIDIA server assets (trees, leaves)")
# chain (2026-11-21): --layout pipe = no mesa; the sequence task's mini half-pipe (export_chain.py pipe.json) at the
# origin, the dressing kept clear of the chain paths (--paths). Runs without Kit under plain usd-core too (.usdvenv).
parser.add_argument("--layout", choices=["mesa", "pipe"], default="mesa")
parser.add_argument("--pipe", default=str(HERE / "chain" / "pipe.json"))
parser.add_argument("--paths", default=None, help="glob of *.kin.npz for the clearance check (default takes/ or chain/)")
# cine (2026-11-22, default classic = unchanged): --look cine = golden-hour light (low warm sun, evening sky HDRI) and a
# few more props behind the (trimmed, pipe_trim.py) half-pipe, all kept clear of the paths like the rest
parser.add_argument("--look", choices=["classic", "cine"], default="classic")
args, _ = parser.parse_known_args()

try:                                  # Isaac Sim python; plain usd-core (no renderer, no GPU) also works
    from isaacsim import SimulationApp  # noqa: E402
    app = SimulationApp({"headless": True})
except ImportError:
    app = None

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

PIPE_LAYOUT = args.layout == "pipe"
if PIPE_LAYOUT and Path(args.out).resolve() == (PROJECT / "scenes" / "skatepark" / "park.usda").resolve():
    args.out = str(PROJECT / "scenes" / "skatepark_pipe" / "park.usda")     # never overwrite the mesa park
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
if PIPE_LAYOUT:
    PIPE = json.load(open(args.pipe))
    PARK = dict(height=0.0, half_size=0.0, profiles={})
else:
    PARK = json.load(open(args.park))
H, P = PARK["height"], PARK["half_size"]

# every recorded robot / board xy, for the clearance check of the dressing
PATHS = []
for f in glob.glob(args.paths or str(HERE / ("chain" if PIPE_LAYOUT else "takes") / "*.kin.npz")):
    z = np.load(f)
    PATHS.append(z["root_pos"][:, :2])
    PATHS.append(z["board_body_pos"][:, 0, :2])
PATHS = np.concatenate(PATHS) if PATHS else np.zeros((0, 2))


def clear_of_paths(cx, cy, hx, hy, yaw=0.0, margin=None):
    """Footprint rectangle (centre, half sizes, yaw) at least `margin` from every recorded path point."""
    margin = args.clearance if margin is None else margin
    if not len(PATHS):
        return True
    c, s = math.cos(yaw), math.sin(yaw)
    d = PATHS - np.array([cx, cy])
    u, v = d @ np.array([c, s]), d @ np.array([-s, c])
    dx, dy = np.maximum(np.abs(u) - hx, 0), np.maximum(np.abs(v) - hy, 0)
    return float(np.min(np.hypot(dx, dy))) >= margin


stage = Usd.Stage.CreateNew(args.out) if not os.path.exists(args.out) else None
if stage is None:
    os.remove(args.out)
    stage = Usd.Stage.CreateNew(args.out)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
UsdGeom.Scope.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Park")

# ── materials ────────────────────────────────────────────────────────────────
MATS = {}


def _tex(name, *cands):
    for c in cands:
        p = TEX / name / f"{name}_{c}.png"
        if p.exists():
            return str(p).replace("\\", "/")
    return None


def omnipbr(key, base=None, normal=None, orm=None, tint=(1, 1, 1), rough=0.6, metal=0.0, color=None, bump=1.0,
            opacity=None, emissive=None, spec=0.5, albedo_brightness=1.0):
    if key in MATS:
        return MATS[key]
    path = f"/World/Looks/{key}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateIdAttr("OmniPBR") if False else None
    inp = lambda n, t, v: sh.CreateInput(n, t).Set(v)
    if base:
        inp("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(base))
        sh.GetInput("diffuse_texture").GetAttr().SetColorSpace("sRGB")
    if color is not None:
        inp("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color))
    inp("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*tint))
    inp("albedo_brightness", Sdf.ValueTypeNames.Float, float(albedo_brightness))
    if normal:
        inp("normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(normal))
        sh.GetInput("normalmap_texture").GetAttr().SetColorSpace("raw")
        inp("bump_factor", Sdf.ValueTypeNames.Float, float(bump))
    if orm:
        inp("ORM_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(orm))
        sh.GetInput("ORM_texture").GetAttr().SetColorSpace("raw")
        inp("enable_ORM_texture", Sdf.ValueTypeNames.Bool, True)
        inp("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.0)
        inp("metallic_texture_influence", Sdf.ValueTypeNames.Float, 1.0 if metal > 0 else 0.0)
    else:
        inp("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 0.0)
        inp("metallic_texture_influence", Sdf.ValueTypeNames.Float, 0.0)
    inp("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rough))
    inp("metallic_constant", Sdf.ValueTypeNames.Float, float(metal))
    inp("specular_level", Sdf.ValueTypeNames.Float, float(spec))
    if opacity:
        inp("enable_opacity", Sdf.ValueTypeNames.Bool, True)
        inp("enable_opacity_texture", Sdf.ValueTypeNames.Bool, True)
        inp("opacity_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(opacity))
        sh.GetInput("opacity_texture").GetAttr().SetColorSpace("raw")
        inp("opacity_mode", Sdf.ValueTypeNames.Int, 0)
        inp("opacity_threshold", Sdf.ValueTypeNames.Float, 0.5)
    if emissive:
        inp("enable_emission", Sdf.ValueTypeNames.Bool, True)
        inp("emissive_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*emissive[0]))
        inp("emissive_intensity", Sdf.ValueTypeNames.Float, float(emissive[1]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    MATS[key] = mat
    return mat


def base_mat(key, name, **kw):
    return omnipbr(key, base=_tex(name, "BaseColor"), normal=_tex(name, "Normal", "N"), orm=_tex(name, "ORM"), **kw)


GEN = str(TEX / "gen").replace("\\", "/")
M = dict(
    deck=base_mat("ParkConcrete", "Concrete_Smooth", tint=(0.93, 0.92, 0.9)),
    ramp=base_mat("RampConcrete", "Concrete_Smooth", tint=(0.86, 0.86, 0.85)),
    pipe=base_mat("PipeConcrete", "Concrete_Smooth", tint=(0.9, 0.89, 0.86), rough=0.55),
    floor=base_mat("FloorConcrete", "Concrete_Polished", tint=(0.78, 0.77, 0.75), bump=0.6),
    wall=base_mat("WallConcrete", "Concrete_Formed"),
    block=base_mat("Block", "Concrete_Block"),
    asphalt=base_mat("Asphalt", "Asphalt", tint=(0.8, 0.8, 0.8)),
    grass=base_mat("Grass", "Grass_Cut"),
    steel=base_mat("Steel", "Steel_Stainless", metal=1.0),
    galv=base_mat("Galvanized", "Steel_Cast", metal=1.0, tint=(0.85, 0.86, 0.88)),
    rust=base_mat("Rust", "RustedMetal", metal=1.0),
    wood=base_mat("BenchWood", "Oak"),
    ply=base_mat("Plywood", "Plywood"),
    joint=omnipbr("Joint", color=(0.12, 0.12, 0.12), rough=0.9),
    paint_y=omnipbr("PaintYellow", color=(0.85, 0.62, 0.05), rough=0.55),
    paint_r=omnipbr("PaintRed", color=(0.7, 0.08, 0.06), rough=0.5),
    cone=omnipbr("ConeOrange", color=(0.95, 0.32, 0.03), rough=0.45),
    white=omnipbr("ConeWhite", color=(0.85, 0.85, 0.85), rough=0.4),
    black=omnipbr("BlackPlastic", color=(0.03, 0.03, 0.035), rough=0.55),
    mural=omnipbr("Mural", base=GEN + "/mural.png", rough=0.85),
    fence=omnipbr("ChainLink", base=GEN + "/chainlink.png", opacity=GEN + "/chainlink.png", metal=1.0, rough=0.45,
                  color=(0.65, 0.66, 0.66)),
    lamp=omnipbr("LampGlass", color=(1, 0.95, 0.85), emissive=((1.0, 0.92, 0.8), 2000.0), rough=0.3),
)


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# ── mesh helpers ─────────────────────────────────────────────────────────────
class MeshBuilder:
    """Flat-shaded polygon soup with faceVarying UVs (meters / tile) and normals."""

    def __init__(self, tile=2.0):
        self.pts, self.counts, self.idx, self.st, self.nrm = [], [], [], [], []
        self.tile = tile

    def poly(self, verts, uv=None):
        verts = [np.asarray(v, float) for v in verts]
        n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        if np.linalg.norm(n) < 1e-12 and len(verts) > 3:
            n = np.cross(verts[2] - verts[0], verts[3] - verts[0])
        n = n / max(np.linalg.norm(n), 1e-12)
        if uv is None:   # planar projection on the dominant axis plane
            a = int(np.argmax(np.abs(n)))
            ax = [i for i in range(3) if i != a]
            uv = [(v[ax[0]], v[ax[1]]) for v in verts]
        base = len(self.pts)
        self.pts += [tuple(v) for v in verts]
        self.counts.append(len(verts))
        self.idx += list(range(base, base + len(verts)))
        self.st += [(u / self.tile, w / self.tile) for u, w in uv]
        self.nrm += [tuple(n)] * len(verts)

    def box(self, center, half, yaw=0.0, pitch=0.0):
        c = np.asarray(center, float)
        hx, hy, hz = half
        R = rot(yaw, pitch)
        corners = [c + R @ np.array([sx * hx, sy * hy, sz * hz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        k = lambda sx, sy, sz: corners[(sx > 0) * 4 + (sy > 0) * 2 + (sz > 0)]
        faces = [
            [k(-1, -1, 1), k(1, -1, 1), k(1, 1, 1), k(-1, 1, 1)],
            [k(-1, -1, -1), k(-1, 1, -1), k(1, 1, -1), k(1, -1, -1)],
            [k(-1, -1, -1), k(1, -1, -1), k(1, -1, 1), k(-1, -1, 1)],
            [k(-1, 1, -1), k(-1, 1, 1), k(1, 1, 1), k(1, 1, -1)],
            [k(-1, -1, -1), k(-1, -1, 1), k(-1, 1, 1), k(-1, 1, -1)],
            [k(1, -1, -1), k(1, 1, -1), k(1, 1, 1), k(1, -1, 1)],
        ]
        for f in faces:
            self.poly(f)

    def cylinder(self, p0, p1, r, n=16, caps=True):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = p1 - p0
        L = np.linalg.norm(ax)
        ax /= L
        t = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(ax, t)
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
        for i in range(n):
            (c0, s0), (c1, s1) = ring[i], ring[i + 1]
            a0, a1 = r * (c0 * u + s0 * v), r * (c1 * u + s1 * v)
            arc0, arc1 = 2 * math.pi * r * i / n, 2 * math.pi * r * (i + 1) / n
            self.poly([p0 + a0, p0 + a1, p1 + a1, p1 + a0], uv=[(arc0, 0), (arc1, 0), (arc1, L), (arc0, L)])
        if caps:
            self.poly([p1 + r * (c * u + s * v) for c, s in ring[:-1]])
            self.poly([p0 + r * (c * u + s * v) for c, s in reversed(ring[:-1])])

    def emit(self, path, mat, smooth=False):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in self.pts]))
        mesh.CreateFaceVertexCountsAttr(self.counts)
        mesh.CreateFaceVertexIndicesAttr(self.idx)
        mesh.CreateSubdivisionSchemeAttr("none")
        if not smooth:
            mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in self.nrm]))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*s) for s in self.st]))
        pts = np.array(self.pts)
        mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))])
        bind(mesh.GetPrim(), mat)
        return mesh


def rot(yaw=0.0, pitch=0.0):
    cy, sy, cp, sp_ = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp_], [0, 1, 0], [-sp_, 0, cp]])
    return Rz @ Ry


# ── the MuJoCo park: mesa top, ramps, ramp side walls ───────────────────────
def build_mesa():
    top = MeshBuilder(tile=2.5)
    top.poly([(-P, -P, H), (P, -P, H), (P, P, H), (-P, P, H)])
    top.emit("/World/Park/Mesa/Top", M["deck"])
    ramps = MeshBuilder(tile=2.5)
    walls = MeshBuilder(tile=2.0)
    edges = MeshBuilder(tile=0.5)
    for side, prof in PARK["profiles"].items():
        psi = math.radians(prof["yaw_deg"])
        c, s = math.cos(psi), math.sin(psi)
        d, zz = np.array(prof["d"]), np.array(prof["z"])
        W = lambda u, w, z: (u * c - w * s, u * s + w * c, z)
        for i in range(len(d) - 1):
            u0, u1 = P + d[i], P + d[i + 1]
            z0, z1 = zz[i], zz[i + 1]
            # surface quad, UV in world-ish (u along the ramp, w across)
            ramps.poly([W(u0, -P, z0), W(u1, -P, z1), W(u1, P, z1), W(u0, P, z0)],
                       uv=[(u0, -P), (u1, -P), (u1, P), (u0, P)])
            for w, sg in ((-P, -1), (P, 1)):
                if max(z0, z1) < 1e-4:
                    continue
                quad = [W(u0, w, 0.0), W(u1, w, 0.0), W(u1, w, z1), W(u0, w, z0)]
                if sg > 0:
                    quad = quad[::-1]
                walls.poly(quad, uv=[(u0, 0), (u1, 0), (u1, z1), (u0, z0)])
                # steel angle on the side wall top edge (flush with the surface, 1 mm proud of the wall)
                e = 0.0012 * sg
                hgt = min(0.035, max(z0, z1))
                q2 = [W(u0, w + e, max(z0 - hgt, 0)), W(u1, w + e, max(z1 - hgt, 0)), W(u1, w + e, z1),
                      W(u0, w + e, z0)]
                if sg > 0:
                    q2 = q2[::-1]
                edges.poly(q2, uv=[(u0, 0), (u1, 0), (u1, 0.03), (u0, 0.03)])
    ramps.emit("/World/Park/Mesa/Ramps", M["ramp"])
    walls.emit("/World/Park/Mesa/SideWalls", M["wall"])
    edges.emit("/World/Park/Mesa/SteelEdges", M["galv"])


def build_pipe():
    """The sequence task's mini half-pipe, exactly the surface MuJoCo collides with (pipe.json: the upper envelope of
    the terrain box facets, a polyline in x, extruded over y0..y1). Transition facets are the true 4 deg chords; their
    normals are smoothed so the concrete reads as one curve. The flat bottom is lifted 0.4 mm off the pad (no
    z-fight; the wheels sink ~0.4 mm into MuJoCo's soft contacts anyway). Solid end faces, back walls, steel coping."""
    px, pz = np.array(PIPE["profile_x"], float), np.array(PIPE["profile_z"], float)
    keep = [0] + [i for i in range(1, len(px)) if abs(px[i] - px[i - 1]) > 1e-7 or abs(pz[i] - pz[i - 1]) > 1e-7]
    px, pz = px[keep], pz[keep]
    pz = np.where(pz < 1e-6, 4e-4, pz)
    y0, y1 = PIPE["y0"], PIPE["y1"]
    h_deck = float(np.max(pz[np.abs(px) > PIPE["x_hi"] - 0.1]))               # platform top
    # riding surface: polyline between the two platform back edges (drop the vertical outer faces)
    on = pz >= h_deck - 1e-4
    i0, i1 = int(np.argmax(on)), len(pz) - 1 - int(np.argmax(on[::-1]))
    sx, sz = px[i0:i1 + 1], pz[i0:i1 + 1]
    seg = np.stack([np.diff(sx), np.diff(sz)], 1)
    L = np.linalg.norm(seg, axis=1)
    ok = L > 1e-7
    sx, sz = np.r_[sx[0], sx[1:][ok]], np.r_[sz[0], sz[1:][ok]]
    seg = np.stack([np.diff(sx), np.diff(sz)], 1)
    L = np.linalg.norm(seg, axis=1)
    sn = np.stack([-seg[:, 1], seg[:, 0]], 1) / L[:, None]               # upward normals (x, z)
    arc = np.r_[0.0, np.cumsum(L)]

    def vnorm(j, s):        # normal at vertex j of segment s: smoothed with the neighbour below 12 deg of turn
        o = s - 1 if j == s else s + 1
        if 0 <= o < len(sn) and sn[o] @ sn[s] > math.cos(math.radians(12)):
            v = sn[o] + sn[s]
            return v / np.linalg.norm(v)
        return sn[s]
    pts, nrm, st, cnt, idx = [], [], [], [], []
    ny = max(2, int(round((y1 - y0) / 0.5)))
    yy = np.linspace(y0, y1, ny + 1)
    tile = 1.2
    for s in range(len(seg)):
        for k in range(ny):
            quad = [(s, yy[k]), (s + 1, yy[k]), (s + 1, yy[k + 1]), (s, yy[k + 1])]
            for (j, y) in quad:
                n2 = vnorm(j, s)
                pts.append(Gf.Vec3f(float(sx[j]), float(y), float(sz[j])))
                nrm.append(Gf.Vec3f(float(n2[0]), 0.0, float(n2[1])))
                st.append(Gf.Vec2f(float(arc[j] / tile), float(y / tile)))
                idx.append(len(pts) - 1)
            cnt.append(4)
    # winding: quad (x0,y0)->(x1,y0)->(x1,y1)->(x0,y1) with x increasing has normal +z for a rising profile: ok
    mesh = UsdGeom.Mesh.Define(stage, "/World/Park/HalfPipe/Surface")
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(cnt)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateNormalsAttr(Vt.Vec3fArray(nrm))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray(st))
    P3 = np.array([[p[0], p[1], p[2]] for p in pts])
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, P3.min(0))), Gf.Vec3f(*map(float, P3.max(0)))])
    bind(mesh.GetPrim(), M["pipe"])
    # end faces (y0, y1) and the outer back walls: formed concrete, solid down to the pad
    walls = MeshBuilder(tile=0.8)
    for y, flip in ((y0, False), (y1, True)):
        for s in range(len(seg)):
            if max(sz[s], sz[s + 1]) < 1e-3:
                continue
            q = [(sx[s], y, 0.0), (sx[s + 1], y, 0.0), (sx[s + 1], y, sz[s + 1]), (sx[s], y, sz[s])]
            walls.poly(q[::-1] if flip else q, uv=[(sx[s], 0), (sx[s + 1], 0), (sx[s + 1], sz[s + 1]), (sx[s], sz[s])])
    for x, flip in ((sx[0], True), (sx[-1], False)):
        q = [(x, y0, 0.0), (x, y1, 0.0), (x, y1, h_deck), (x, y0, h_deck)]
        walls.poly(q[::-1] if flip else q, uv=[(y0, 0), (y1, 0), (y1, h_deck), (y0, h_deck)])
    walls.emit("/World/Park/HalfPipe/Walls", M["wall"])
    # steel angle on the end faces' top edge (like the mesa ramps), 1 mm proud of the face
    edges = MeshBuilder(tile=0.5)
    for y, sg in ((y0, -1), (y1, 1)):
        e = 0.001 * sg
        for s in range(len(seg)):
            if max(sz[s], sz[s + 1]) < 2e-3:
                continue
            hgt = min(0.012, max(sz[s], sz[s + 1]))
            q = [(sx[s], y + e, max(sz[s] - hgt, 0)), (sx[s + 1], y + e, max(sz[s + 1] - hgt, 0)),
                 (sx[s + 1], y + e, sz[s + 1]), (sx[s], y + e, sz[s])]
            edges.poly(q if sg > 0 else q[::-1], uv=[(sx[s], 0), (sx[s + 1], 0), (sx[s + 1], 0.01), (sx[s], 0.01)])
    edges.emit("/World/Park/HalfPipe/SteelEdges", M["galv"])
    # coping: a steel tube along each lip (inner platform edge), its top 5 mm above the platform; it covers only the
    # top ~1.5 cm of the 12 cm wall (the chain's passes rise 1-6 cm)
    cop = MeshBuilder(tile=0.3)
    r = 0.0065
    for sg in (-1, 1):
        xl = sg * min(abs(x) for x in sx[(np.abs(sx) > 0.3) & (sz >= h_deck - 1e-4)])   # inner edge at deck height
        cop.cylinder((xl + sg * 0.0035, y0, h_deck - 0.0015), (xl + sg * 0.0035, y1, h_deck - 0.0015), r, n=20)
    cop.emit("/World/Park/HalfPipe/Coping", M["steel"], smooth=True)
    print(f"PARK half-pipe: {len(seg)} profile segments, x {sx[0]:.3f}..{sx[-1]:.3f}, y {y0:.2f}..{y1:.2f}, "
          f"platform {h_deck:.3f} m", flush=True)


def in_park(x, y, pad=0.02):
    if PIPE_LAYOUT:
        return PIPE["x_lo"] - pad <= x <= PIPE["x_hi"] + pad and PIPE["y0"] - pad <= y <= PIPE["y1"] + pad
    if abs(x) <= P + pad and abs(y) <= P + pad:
        return True
    for side, prof in PARK["profiles"].items():
        psi = math.radians(prof["yaw_deg"])
        u = x * math.cos(psi) + y * math.sin(psi)
        w = -x * math.sin(psi) + y * math.cos(psi)
        if P - pad <= u <= P + prof["d"][-1] + pad and abs(w) <= P + pad:
            return True
    return False


# ── floor: pad, expansion joints, curb, outside ─────────────────────────────
PAD = (-20.0, 24.0, -20.0, 16.0)


def build_floor():
    x0, x1, y0, y1 = PAD
    f = MeshBuilder(tile=3.0)
    f.poly([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])
    f.emit("/World/Park/Floor/Pad", M["floor"])
    j = MeshBuilder(tile=1.0)
    hw, step = 0.006, 0.5
    for gx in np.arange(x0 + 3, x1, 3.0):           # joints along y at x = gx
        for yy in np.arange(y0, y1, step):
            if not (in_park(gx, yy) or in_park(gx, yy + step)):
                j.poly([(gx - hw, yy, 4e-4), (gx + hw, yy, 4e-4), (gx + hw, yy + step, 4e-4), (gx - hw, yy + step, 4e-4)])
    for gy in np.arange(y0 + 3, y1, 3.0):
        for xx in np.arange(x0, x1, step):
            if not (in_park(xx, gy) or in_park(xx + step, gy)):
                j.poly([(xx, gy - hw, 4e-4), (xx + step, gy - hw, 4e-4), (xx + step, gy + hw, 4e-4), (xx, gy + hw, 4e-4)])
    j.emit("/World/Park/Floor/Joints", M["joint"])
    # outside: asphalt (slightly below, never coplanar) and a grass verge
    a = MeshBuilder(tile=4.0)
    R = 400.0
    a.poly([(-R, -R, -0.01), (R, -R, -0.01), (R, R, -0.01), (-R, R, -0.01)])
    a.emit("/World/Park/Outside/Asphalt", M["asphalt"])
    g = MeshBuilder(tile=2.0)
    for (gx0, gx1, gy0, gy1) in ((x0 - 6, x1 + 6, y1 + 0.3, y1 + 6), (x0 - 6, x1 + 6, y0 - 6, y0 - 0.3),
                                 (x0 - 6, x0 - 0.3, y0 - 0.3, y1 + 0.3), (x1 + 0.3, x1 + 6, y0 - 0.3, y1 + 0.3)):
        g.poly([(gx0, gy0, 0.005), (gx1, gy0, 0.005), (gx1, gy1, 0.005), (gx0, gy1, 0.005)])
    g.emit("/World/Park/Outside/Grass", M["grass"])
    c = MeshBuilder(tile=1.0)
    t = 0.25
    for (cx, cy, hx, hy) in (((x0 + x1) / 2, y0 - t / 2, (x1 - x0) / 2 + t, t / 2), ((x0 + x1) / 2, y1 + t / 2, (x1 - x0) / 2 + t, t / 2),
                             (x0 - t / 2, (y0 + y1) / 2, t / 2, (y1 - y0) / 2), (x1 + t / 2, (y0 + y1) / 2, t / 2, (y1 - y0) / 2)):
        c.box((cx, cy, 0.06), (hx, hy, 0.07))
    c.emit("/World/Park/Floor/Curb", M["wall"])


# ── dressing ─────────────────────────────────────────────────────────────────
PLACED = []


def place_ok(name, cx, cy, hx, hy, yaw=0.0, margin=None):
    ok = clear_of_paths(cx, cy, hx, hy, yaw, margin) and not in_park(cx, cy, pad=max(hx, hy) + 0.3)
    PLACED.append((name, ok))
    if not ok:
        print(f"PARK skip {name}: too close to a recorded path or the mesa", flush=True)
    return ok


def quarter_pipe(name, cx, cy, yaw, width=8.0, height=1.2, radius=1.8, deck=1.2):
    """Transition facing local -x (riders come from -x), coping pipe, deck, back wall."""
    if not place_ok(name, cx + 0.5 * (radius + deck) * math.cos(yaw), cy + 0.5 * (radius + deck) * math.sin(yaw),
                    0.5 * (radius + deck) + 0.2, width / 2, yaw):
        return
    R = rot(yaw)
    o = np.array([cx, cy, 0.0])
    Wp = lambda x, y, z: tuple(o + R @ np.array([x, y, z]))
    n = 24
    ang0 = math.acos(1 - height / radius)
    surf = MeshBuilder(tile=2.5)
    prof = []
    for i in range(n + 1):
        a = ang0 * i / n
        prof.append((radius * math.sin(a), radius * (1 - math.cos(a))))
    for i in range(n):
        (x0, z0), (x1, z1) = prof[i], prof[i + 1]
        arc0, arc1 = radius * ang0 * i / n, radius * ang0 * (i + 1) / n
        surf.poly([Wp(x0, -width / 2, z0), Wp(x0, width / 2, z0), Wp(x1, width / 2, z1), Wp(x1, -width / 2, z1)],
                  uv=[(arc0, -width / 2), (arc0, width / 2), (arc1, width / 2), (arc1, -width / 2)])
    xt, zt = prof[-1]
    surf.poly([Wp(xt, -width / 2, height), Wp(xt, width / 2, height), Wp(xt + deck, width / 2, height),
               Wp(xt + deck, -width / 2, height)])
    surf.emit(f"/World/Park/{name}/Surface", M["ramp"])
    sides = MeshBuilder(tile=2.0)
    for y, flip in ((-width / 2, False), (width / 2, True)):
        poly = [Wp(0, y, 0)] + [Wp(x, y, z) for x, z in prof] + [Wp(xt + deck, y, height), Wp(xt + deck, y, 0)]
        sides.poly(poly[::-1] if not flip else poly)
    sides.poly([Wp(xt + deck, -width / 2, 0), Wp(xt + deck, -width / 2, height), Wp(xt + deck, width / 2, height),
                Wp(xt + deck, width / 2, 0)])
    sides.emit(f"/World/Park/{name}/Walls", M["wall"])
    cop = MeshBuilder(tile=0.3)
    cop.cylinder(Wp(xt + 0.02, -width / 2, height + 0.005), Wp(xt + 0.02, width / 2, height + 0.005), 0.03, n=16)
    cop.emit(f"/World/Park/{name}/Coping", M["steel"])
    rail = MeshBuilder(tile=0.5)
    for y in np.linspace(-width / 2 + 0.3, width / 2 - 0.3, 5):
        rail.cylinder(Wp(xt + deck - 0.1, y, height), Wp(xt + deck - 0.1, y, height + 0.9), 0.025, n=10)
    rail.cylinder(Wp(xt + deck - 0.1, -width / 2 + 0.3, height + 0.9), Wp(xt + deck - 0.1, width / 2 - 0.3, height + 0.9),
                  0.025, n=10)
    rail.emit(f"/World/Park/{name}/Rail", M["galv"])


def ledge(name, cx, cy, yaw, L=4.0, Wd=0.55, Ht=0.42):
    if not place_ok(name, cx, cy, L / 2, Wd / 2, yaw):
        return
    b = MeshBuilder(tile=1.5)
    b.box((cx, cy, Ht / 2), (L / 2, Wd / 2, Ht / 2), yaw)
    b.emit(f"/World/Park/{name}/Block", M["wall"])
    e = MeshBuilder(tile=0.3)
    R = rot(yaw)
    for sg in (-1, 1):
        p0 = np.array([cx, cy, Ht]) + R @ np.array([-L / 2, sg * Wd / 2, 0])
        p1 = np.array([cx, cy, Ht]) + R @ np.array([L / 2, sg * Wd / 2, 0])
        e.cylinder(p0, p1, 0.012, n=8)
    e.emit(f"/World/Park/{name}/Edge", M["galv"])
    p = MeshBuilder(tile=1.0)
    p.box((cx, cy, Ht - 0.0), (L / 2 - 0.02, Wd / 2 - 0.02, 0.003), yaw)
    p.emit(f"/World/Park/{name}/Paint", M["paint_y"])


def flat_rail(name, cx, cy, yaw, L=4.5, Ht=0.32):
    if not place_ok(name, cx, cy, L / 2, 0.2, yaw):
        return
    R = rot(yaw)
    r = MeshBuilder(tile=0.3)
    p = lambda x, z: tuple(np.array([cx, cy, z]) + R @ np.array([x, 0, 0]))
    r.cylinder(p(-L / 2, Ht), p(L / 2, Ht), 0.025, n=16)
    for x in (-L / 2 + 0.25, L / 2 - 0.25):
        r.cylinder(p(x, 0), p(x, Ht), 0.02, n=10)
        r.box(p(x, 0.005), (0.12, 0.08, 0.005), yaw)
    r.emit(f"/World/Park/{name}/Rail", M["paint_r"])


def stairs(name, cx, cy, yaw, n=3, rise=0.15, run=0.35, W=3.0, top=2.0):
    total = n * run + top
    if not place_ok(name, cx + 0.5 * total * math.cos(yaw), cy + 0.5 * total * math.sin(yaw), total / 2, W / 2, yaw):
        return
    R = rot(yaw)
    b = MeshBuilder(tile=1.5)
    for i in range(n):
        x0 = i * run
        hz = (i + 1) * rise
        cxi = x0 + (total - x0) / 2
        b.box(tuple(np.array([cx, cy, 0]) + R @ np.array([cxi, 0, hz / 2])), ((total - x0) / 2, W / 2, hz / 2), yaw)
    b.emit(f"/World/Park/{name}/Steps", M["wall"])
    hr = MeshBuilder(tile=0.3)
    P0 = np.array([cx, cy, 0]) + R @ np.array([-0.1, 0, 0.85])
    P1 = np.array([cx, cy, 0]) + R @ np.array([n * run + 0.1, 0, n * rise + 0.85])
    hr.cylinder(P0, P1, 0.022, n=12)
    for t in (0.0, 1.0):
        q = P0 + t * (P1 - P0)
        hr.cylinder(q - np.array([0, 0, 0.85]), q, 0.02, n=10)
    hr.emit(f"/World/Park/{name}/Handrail", M["galv"])


def mural_wall(name, cx, cy, yaw, L=12.0, Ht=2.6, T=0.3):
    if not place_ok(name, cx, cy, L / 2, T / 2, yaw):
        return
    b = MeshBuilder(tile=2.0)
    b.box((cx, cy, Ht / 2), (L / 2, T / 2, Ht / 2), yaw)
    b.emit(f"/World/Park/{name}/Wall", M["block"])
    R = rot(yaw)
    m = MeshBuilder(tile=1.0)
    face = lambda x, z: tuple(np.array([cx, cy, 0]) + R @ np.array([x, -T / 2 - 0.002, z]))
    m.poly([face(-L / 2, 0.0), face(L / 2, 0.0), face(L / 2, Ht), face(-L / 2, Ht)],
           uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
    m.emit(f"/World/Park/{name}/Mural", M["mural"])


def fence(name, pts, Ht=2.4, post_every=3.0):
    f = MeshBuilder(tile=1.0)
    posts = MeshBuilder(tile=0.5)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        # mesh fabric: UV tile = 0.24 m diamonds (texture holds 4 cells over 1 tile -> 1 tile = 0.24*4 m)
        tile = 0.24
        f.poly([(x0, y0, 0.05), (x1, y1, 0.05), (x1, y1, Ht), (x0, y0, Ht)],
               uv=[(0, 0), (L / tile, 0), (L / tile, (Ht - 0.05) / tile), (0, (Ht - 0.05) / tile)])
        k = max(1, int(round(L / post_every)))
        for i in range(k + 1):
            t = i / k
            px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            posts.cylinder((px, py, 0), (px, py, Ht + 0.05), 0.03, n=10)
        posts.cylinder((x0, y0, Ht), (x1, y1, Ht), 0.02, n=8, caps=False)
    f.tile = 1.0
    f.emit(f"/World/Park/{name}/Fabric", M["fence"])
    posts.emit(f"/World/Park/{name}/Posts", M["galv"])


def lamp_post(name, x, y, Ht=6.0, yaw=0.0):
    if not place_ok(name, x, y, 0.3, 0.3):
        return
    p = MeshBuilder(tile=0.5)
    p.cylinder((x, y, 0), (x, y, Ht), 0.07, n=12)
    hx, hy = x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw)
    p.cylinder((x, y, Ht - 0.05), (hx, hy, Ht - 0.05), 0.04, n=8)
    p.box((hx, hy, Ht - 0.1), (0.25, 0.12, 0.06), yaw)
    p.emit(f"/World/Park/{name}/Pole", M["galv"])
    g = MeshBuilder(tile=0.5)
    g.box((hx, hy, Ht - 0.165), (0.22, 0.1, 0.005), yaw)
    g.emit(f"/World/Park/{name}/Glass", M["lamp"])


def bench(name, x, y, yaw):
    if not place_ok(name, x, y, 1.0, 0.3, yaw):
        return
    R = rot(yaw)
    w = MeshBuilder(tile=1.0)
    c = MeshBuilder(tile=1.0)
    P_ = lambda dx, dy, dz: tuple(np.array([x, y, dz]) + R @ np.array([dx, dy, 0]))
    for dx in (-0.8, 0.8):
        c.box(P_(dx, 0, 0.21), (0.08, 0.25, 0.21), yaw)
    for dy in (-0.15, 0.0, 0.15):
        w.box(P_(0, dy, 0.44), (1.0, 0.06, 0.02), yaw)
    c.emit(f"/World/Park/{name}/Legs", M["wall"])
    w.emit(f"/World/Park/{name}/Slats", M["wood"])


def cone(name, x, y, Ht=0.46, r=0.13):
    if not place_ok(name, x, y, 0.2, 0.2, margin=0.5):
        return
    n = 20
    mb = MeshBuilder(tile=0.3)
    wb = MeshBuilder(tile=0.3)
    mb.box((x, y, 0.012), (0.19, 0.19, 0.012))
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        for (z0, z1, tgt) in ((0.025, 0.25, mb), (0.25, 0.33, wb), (0.33, Ht, mb)):
            r0 = r * (1 - (z0 - 0.025) / (Ht - 0.025)) + 0.015
            r1 = r * (1 - (z1 - 0.025) / (Ht - 0.025)) + 0.015
            tgt.poly([(x + r0 * math.cos(a0), y + r0 * math.sin(a0), z0), (x + r0 * math.cos(a1), y + r0 * math.sin(a1), z0),
                      (x + r1 * math.cos(a1), y + r1 * math.sin(a1), z1), (x + r1 * math.cos(a0), y + r1 * math.sin(a0), z1)])
    mb.emit(f"/World/Park/{name}/Body", M["cone"], smooth=True)
    wb.emit(f"/World/Park/{name}/Stripe", M["white"], smooth=True)


def trash_can(name, x, y):
    if not place_ok(name, x, y, 0.35, 0.35):
        return
    t = MeshBuilder(tile=0.5)
    t.cylinder((x, y, 0), (x, y, 0.95), 0.3, n=20)
    t.emit(f"/World/Park/{name}/Can", M["rust"])


# ── remote assets (trees, leaves, rocks) ─────────────────────────────────────
def remote(name, url, x, y, yaw_deg=0.0, scale=1.0, check=True, hx=0.5):
    if check and not place_ok(name, x, y, hx, hx):
        return
    try:
        ref = Usd.Stage.Open(url)
        mpu = UsdGeom.GetStageMetersPerUnit(ref) or 0.01
        up = UsdGeom.GetStageUpAxis(ref)
    except Exception as e:  # noqa: BLE001
        if app is not None:
            print(f"PARK remote {name} failed: {e}", flush=True)
            return
        mpu, up = 0.01, UsdGeom.Tokens.z      # usd-core: no omni client; the NVIDIA vegetation is cm, z up

    xf = UsdGeom.Xform.Define(stage, f"/World/Park/Dressing/{name}")
    UsdGeom.Xform.Define(stage, f"/World/Park/Dressing/{name}/Asset").GetPrim().GetReferences().AddReference(url)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0))
    xf.AddRotateZOp().Set(float(yaw_deg))
    if up == UsdGeom.Tokens.y:
        xf.AddRotateXOp().Set(90.0)
    xf.AddScaleOp().Set(Gf.Vec3f(float(mpu * scale)))


def build_dressing():
    x0, x1, y0, y1 = PAD
    quarter_pipe("QuarterPipeEast", 18.5, -4.0, 0.0, width=10.0)
    quarter_pipe("QuarterPipeNorth", 2.0, 11.8, math.pi / 2, width=10.0, height=1.0, radius=1.6)
    ledge("LedgeNW", -12.0, 8.0, math.radians(15))
    ledge("LedgeSE", 17.0, -15.5, math.radians(90), L=3.5)
    flat_rail("FlatRail", -1.0, -14.5, 0.0)
    stairs("Stairs", -18.6, -12.0, 0.0, W=3.5)
    mural_wall("MuralNorth", -12.0, 15.6, 0.0, L=14.0)
    mural_wall("MuralSouth", 6.0, -19.6, math.pi, L=14.0)
    fence("Fence", [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
    for i, (lx, ly, ly_yaw) in enumerate(((x0 + 1, y0 + 1, 0.8), (x1 - 1, y0 + 1, 2.4), (x1 - 1, y1 - 1, -2.4),
                                         (x0 + 1, y1 - 1, -0.8), (3.0, y1 - 1, -1.57))):
        lamp_post(f"Lamp{i}", lx, ly, yaw=ly_yaw)
    bench("Bench0", -8.0, -17.5, 0.0)
    bench("Bench1", 10.0, 14.0, math.pi)
    trash_can("Trash0", -5.6, -17.6)
    for i, (cx, cy) in enumerate(((10.5, -2.0), (14.5, -7.5), (9.0, -13.5), (-6.5, -3.0), (-11.5, 2.5), (15.5, 2.0))):
        cone(f"Cone{i}", cx, cy)
    if args.no_remote:
        return
    trees = ["Sycamore", "Red_Maple", "Honey_Locust", "White_Ash", "Black_Oak", "Sugar_Maple", "Lombardy_Poplar"]
    rng = np.random.default_rng(5)
    k = 0
    for (tx0, tx1, ty0, ty1, n) in ((x0 - 14, x1 + 14, y1 + 3, y1 + 12, 9), (x0 - 14, x1 + 14, y0 - 12, y0 - 3, 9),
                                    (x0 - 12, x0 - 3, y0, y1, 5), (x1 + 3, x1 + 12, y0, y1, 5)):
        for _ in range(n):
            remote(f"Tree{k}", f"{S3}/Vegetation/Trees/{trees[k % len(trees)]}.usd", float(rng.uniform(tx0, tx1)),
                   float(rng.uniform(ty0, ty1)), float(rng.uniform(0, 360)), check=False)
            k += 1
    for i in range(10):
        lx, ly = float(rng.uniform(x0 + 1, x1 - 1)), float(rng.uniform(y0 + 1, y1 - 1))
        if in_park(lx, ly, pad=0.5) or not clear_of_paths(lx, ly, 0.6, 0.6, margin=0.4):
            continue
        remote(f"Leaves{i}", f"{S3}/Vegetation/Debris/{['fallcluster1', 'fallcluster2', 'maplefall1', 'oakfall1'][i % 4]}.usd",
               lx, ly, float(rng.uniform(0, 360)), check=False)
    for i in range(8):
        sx, sy = [(x0 + 0.6, float(rng.uniform(y0, y1))), (float(rng.uniform(x0, x1)), y1 - 0.6)][i % 2]
        remote(f"Shrub{i}", f"{S3}/Vegetation/Shrub/{['Boxwood', 'Fountain_Grass_Tall', 'Juniper', 'Switchgrass'][i % 4]}.usd",
               sx, sy, float(rng.uniform(0, 360)), check=True, hx=0.6)


def manual_pad(name, cx, cy, yaw, L=3.0, Wd=1.1, Ht=0.22):
    if not place_ok(name, cx, cy, L / 2, Wd / 2, yaw):
        return
    b = MeshBuilder(tile=1.5)
    b.box((cx, cy, Ht / 2), (L / 2, Wd / 2, Ht / 2), yaw)
    b.emit(f"/World/Park/{name}/Block", M["ramp"])
    e = MeshBuilder(tile=0.3)
    R = rot(yaw)
    for sg in (-1, 1):
        e.cylinder(np.array([cx, cy, Ht]) + R @ np.array([-L / 2, sg * Wd / 2, 0]),
                   np.array([cx, cy, Ht]) + R @ np.array([L / 2, sg * Wd / 2, 0]), 0.012, n=8)
    e.emit(f"/World/Park/{name}/Edge", M["galv"])


def build_cine_dressing():
    """cine: props in the background of the half-pipe shots (behind the short pipe) and beside the approach."""
    manual_pad("CineManualPad", -0.6, 3.2, math.radians(4))
    flat_rail("CineFlatRail", 2.1, 1.4, math.radians(68), L=3.5)
    ledge("CineLedge", -3.4, 0.6, math.radians(-80), L=3.2)
    bench("CineBench", 3.6, -1.2, math.radians(95))
    trash_can("CineTrash", 3.3, 0.6)
    for i, (cx, cy) in enumerate(((-1.9, 0.9), (1.4, -0.4), (2.4, -4.6), (-2.6, -6.8), (1.9, -9.5))):
        cone(f"CineCone{i}", cx, cy)


def build_lights():
    if args.look == "cine":
        # the low sun's mirror hot spot on the polished pad blew out: a little rougher floor (texture 55 % + 0.5)
        fsh = stage.GetPrimAtPath("/World/Looks/FloorConcrete/Shader")
        fsh.GetAttribute("inputs:reflection_roughness_texture_influence").Set(0.55)
        fsh.GetAttribute("inputs:reflection_roughness_constant").Set(0.5)
        # golden hour: evening sky dome (dimmer, so the low sun carries the modelling) + a low warm sun from +x +y,
        # i.e. back / side light for cameras looking +y down the approach and the pipe (long shadows toward camera)
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
        dome.CreateTextureFileAttr(str(ASSETS / "hdri" / os.environ.get("PARK_HDRI", "evening_road_01_4k.hdr"))
                                   .replace("\\", "/"))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(float(os.environ.get("PARK_DOME", "650")))
        UsdGeom.Xformable(dome).AddRotateZOp().Set(float(os.environ.get("PARK_DOME_ROT", "0")))
        sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(float(os.environ.get("PARK_SUN", "2600")))
        sun.CreateAngleAttr(0.8)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.80, 0.60))
        xf = UsdGeom.Xformable(sun)
        xf.AddRotateZOp().Set(float(os.environ.get("PARK_SUN_AZ", "143")))
        xf.AddRotateXOp().Set(float(os.environ.get("PARK_SUN_TILT", "66")))      # 90 - tilt = sun elevation
        return
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    dome.CreateTextureFileAttr(str(ASSETS / "hdri" / "abandoned_parking_4k.hdr").replace("\\", "/"))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(float(os.environ.get("PARK_DOME", "900")))
    UsdGeom.Xformable(dome).AddRotateZOp().Set(0.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(float(os.environ.get("PARK_SUN", "2200")))
    sun.CreateAngleAttr(0.6)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    xf = UsdGeom.Xformable(sun)
    xf.AddRotateZOp().Set(0.0)
    xf.AddRotateXOp().Set(35.0)


build_pipe() if PIPE_LAYOUT else build_mesa()
build_floor()
build_dressing()
if args.look == "cine":
    build_cine_dressing()
build_lights()
stage.GetRootLayer().Save()
print(f"PARK wrote {args.out}: {sum(ok for _, ok in PLACED)}/{len(PLACED)} dressing items placed, "
      f"{len(PATHS)} path points checked", flush=True)
if app is not None:
    app.close()
