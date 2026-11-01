import numpy as np
import pytest
import trimesh
from PIL import Image

from frontend.viewer import build_viewer_html
from reconstruction.mc_shim import marching_cubes_numpy
from reconstruction.mesh_processing import (MeshValidationError, load_mesh, process_mesh, validate_mesh)


def colored_sphere():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    m.visual.vertex_colors = np.c_[(np.abs(m.vertices) * 600).clip(0, 255).astype(np.uint8),
                                   np.full(len(m.vertices), 255, np.uint8)]
    m.apply_translation([2.0, -1.0, 0.5])
    return m


def test_glb_export_roundtrip_keeps_geometry_and_colours(tmp_path):
    res = process_mesh(colored_sphere(), tmp_path)
    assert res.glb_path.exists() and res.obj_path.exists()
    assert res.export_check["readable"] and res.export_check["non_empty"]
    re = load_mesh(res.glb_path)
    assert len(re.faces) == len(res.mesh.faces)
    chk = res.export_check
    assert chk["has_vertex_colors"] and chk["has_normals"] and chk["visual_kind"] == "vertex"
    assert chk["metallic_factors"] and all(m == 0.0 for m in chk["metallic_factors"])   # not the dark glTF default
    assert np.allclose(re.bounds.mean(axis=0), 0, atol=1e-6)        # centred
    assert np.isclose(re.extents.max(), 1.0, atol=1e-6)             # unit size
    assert res.report_after["is_watertight"] and res.report_after["volume"] > 0


def test_inverted_mesh_is_flipped(tmp_path):
    m = colored_sphere()
    m.invert()
    assert m.volume < 0
    res = process_mesh(m, tmp_path)
    assert res.mesh.volume > 0 and any("flipped" in a for a in res.actions)


def test_floaters_removed(tmp_path):
    main = colored_sphere()
    speck = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
    speck.apply_translation([5, 5, 5])
    speck.visual.vertex_colors = [255, 0, 0, 255]
    res = process_mesh(trimesh.util.concatenate([main, speck]), tmp_path)
    assert res.report_after["connected_components"] == 1
    assert any("floating" in a for a in res.actions)


def test_flat_plane_is_rejected(tmp_path):
    plane = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces=[[0, 1, 2], [0, 2, 3]])
    with pytest.raises(MeshValidationError):
        process_mesh(plane, tmp_path)


def test_empty_mesh_is_rejected(tmp_path):
    with pytest.raises(MeshValidationError):
        process_mesh(trimesh.Trimesh(), tmp_path)


def test_validation_reports_degenerate_faces():
    m = trimesh.creation.box()
    v = np.vstack([m.vertices, [[0, 0, 0]]])
    f = np.vstack([m.faces, [[0, 0, 1]]])
    rep = validate_mesh(trimesh.Trimesh(v, f, process=False))
    assert rep["degenerate_faces"] >= 1


def test_texture_is_preserved_in_glb_and_obj(tmp_path):
    box = trimesh.creation.box()
    tex = Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8))
    box.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(1).random((len(box.vertices), 2)),
                                               material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
    res = process_mesh(box, tmp_path)
    assert res.report_after["has_texture"]
    assert load_mesh(res.glb_path).visual.kind == "texture"
    assert res.export_check["has_texture"] and res.export_check["has_uv"]
    assert all(m == 0.0 for m in res.export_check["metallic_factors"])
    assert any(p.suffix == ".png" for p in res.obj_path.parent.iterdir())


def test_marching_cubes_fallback_matches_torchmcubes_convention():
    """Density of an off-centre sphere -> after TripoSR's [2,1,0] reorder, vertices land on the sphere."""
    n = 48
    g = np.linspace(0, 1, n)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    c = np.array([0.6, 0.45, 0.5])
    density = 0.25 - np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    verts, faces = marching_cubes_numpy(density, 0.0)  # TripoSR passes (density - threshold), level 0
    assert len(verts) and len(faces)
    v = verts[:, [2, 1, 0]] / (n - 1)                             # what TripoSR does next
    r = np.linalg.norm(v - c, axis=1)
    assert abs(r.mean() - 0.25) < 0.01 and r.std() < 0.01
    assert np.allclose(v.mean(0), c, atol=0.01)                   # axes not swapped


def test_triposr_orientation_is_a_proper_rotation():
    from reconstruction.image_to_3d import _rot
    R = _rot(np.pi / 2, [0, 1, 0]) @ _rot(-np.pi / 2, [1, 0, 0])
    assert np.isclose(np.linalg.det(R[:3, :3]), 1.0)


def test_viewer_html_embeds_model(tmp_path):
    res = process_mesh(colored_sphere(), tmp_path)
    page = build_viewer_html(res.glb_path, stats={"vertices": 1})
    assert "GLTFLoader" in page and "OrbitControls" in page and len(page) > res.glb_path.stat().st_size


def test_colour_enhancement_matches_reference_brightness():
    from reconstruction.mesh_processing import enhance_vertex_colors
    m = trimesh.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal([60, 15, 15], 12, (len(m.vertices), 3)), 0, 255).astype(np.uint8)
    m.visual.vertex_colors = np.c_[dark, np.full(len(dark), 255, np.uint8)]
    ref = np.zeros((64, 64, 4), np.uint8)
    ref[..., :3] = (200, 40, 35)          # the bright red apple in the generated image
    ref[..., 3] = 255
    ref[..., :3] = np.clip(ref[..., :3] + rng.normal(0, 10, (64, 64, 3)), 0, 255).astype(np.uint8)
    actions = enhance_vertex_colors(m, ref)
    out = np.asarray(m.visual.vertex_colors)[:, :3].astype(float)
    assert out[:, 0].mean() > dark[:, 0].mean() + 80          # brighter red, close to the reference
    assert out[:, :3].std(0).mean() < dark.std(0).mean() * 1.6 + 5   # speckle not amplified
    assert any("colour transfer" in a for a in actions)


def test_smoothing_keeps_volume():
    from reconstruction.mesh_processing import smooth_geometry
    m = trimesh.creation.icosphere(subdivisions=4)
    m.vertices += np.random.default_rng(1).normal(0, 0.01, m.vertices.shape)  # marching-cubes-like noise
    v0 = m.volume
    smooth_geometry(m, 8)
    assert abs(m.volume - v0) / v0 < 0.03
