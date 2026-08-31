# "No reconstruction" mesh export -- experimental change log

Status: **EXPERIMENTAL, UNCOMMITTED**. Nothing in this log has been committed to
git. If this doesn't pan out, `git checkout` the files listed below (or just
`git status` / `git diff` to see exactly what changed) to restore the working
package.

Goal: make "no reconstruction" (reuse the original mesh's own vertex/face
topology, just move the vertices) the **default** export path for deformed
meshes, with the existing Screened Poisson reconstruction becoming an
automatic **fallback** for when the original mesh's faces aren't available
(e.g. the surface points being exported didn't come from that mesh's own
vertices in the first place -- a resampled `create_pcd` point cloud, or points
from some other source entirely).

## Why this touches more than `export_mesh_file`

The existing pipeline's only way to get "surface points" to deform is
`create_pcd()`, which **resamples** a brand new point cloud from the mesh via
Poisson-disk/Monte-Carlo sampling -- those points have no index correspondence
to the mesh's actual vertices or faces, so there's no way to later say "reuse
the original faces" for whatever `create_pcd` handed you. To make "no
reconstruction" possible at all, something upstream of `export_mesh_file` has
to hand back the mesh's *own* vertices and faces, undisturbed, so that the
*same* array (after being pushed through `interpolate_to_surface`) still lines
up with those faces at export time.

That's a new function, not a modification of `create_pcd` (which keeps its
existing resampling behavior/signature untouched -- still useful on its own,
and used as the input to the Poisson fallback path).

## Files changed

### `diffusion_cartogram/core.py`
- **Added** `load_mesh_topology(mesh_path)`: loads a mesh's own vertices,
  faces, and per-vertex normals with **no resampling**. This is the new
  default entry point for the deformation pipeline (replaces `create_pcd` as
  the "get me surface points" call for most users; `create_pcd` still exists
  unchanged for anyone who wants a lighter/resampled point cloud, or as the
  Poisson fallback's input).
- **Modified** `export_mesh_file(filename, deformed_pcd, ...)`:
  - New parameters: `method='none'` (new default), `original_faces=None`.
  - `method='none'`: requires `original_faces`; if given, saves
    `(deformed_pcd, original_faces)` directly via
    `pymeshlab.Mesh(vertex_matrix=..., face_matrix=...)` -- no reconstruction,
    no normal estimation, no Poisson. Guaranteed same topology/watertightness
    as the source mesh, since nothing is being reconstructed.
  - If `method='none'` but `original_faces` is `None`: emits a
    `RuntimeWarning` and **falls back to `method='poisson'`** (the prior,
    only, behavior) automatically.
  - `method='poisson'`: unchanged prior behavior (screened Poisson
    reconstruction with estimated normals), reachable explicitly or via the
    automatic fallback above.
  - Backward compatible: existing callers that only ever passed
    `(filename, deformed_pcd)` keep working exactly as before, just now with
    a printed warning nudging them toward the no-reconstruction path.

### `diffusion_cartogram/__init__.py`
- Exported `load_mesh_topology` alongside the other I/O functions.

### `applications/deform_utils.py`, `applications/run_deformations.py`
- Switched the experiment pipeline's surface-point source from
  `create_pcd(skeleton_cube.stl, n_pts=20_000)` (resampled, ~20-29k points, no
  face correspondence) to `load_mesh_topology(skeleton_cube.stl)` (the mesh's
  actual 69,600 vertices / 139,520 faces).
- `run_experiment()` now threads `faces` through to
  `export_mesh_file(..., method='none', original_faces=faces)`.

## Verified before wiring into the package (see prior turn's empirical tests)

- `pymeshlab.Mesh(vertex_matrix=V, face_matrix=F)` round-trips a translated
  copy of `skeleton_cube.stl`'s own 69,600/139,520 vertex/face arrays back out
  as a **watertight** STL (trivially true -- same topology as the input).
- By contrast, on that same mesh's resampled point cloud, neither ball
  pivoting (radii 1/2/3%) nor alpha shapes (`'Alpha Shape'`/`'Alpha Complex'`,
  alpha 2%/5%) ever produced a watertight result -- motivation for why
  reusing known-good topology, not more reconstruction tuning, is the
  reliable fix for this thin-wireframe geometry.

## Test plan (this turn)

1. `python -m py_compile` the edited package files (syntax only).
2. Small inline test: `load_mesh_topology` + `export_mesh_file(method='none')`
   round-trip, and the `method='none'` -> warns -> falls back to Poisson path
   when `original_faces` is omitted.
3. `run_deformations.py --smoke --only <all prefixes>` with the new pipeline
   wired in, to catch integration breakage cheaply across every experiment
   *type* (sphere, octant BC variants incl. the custom asymmetric-box case,
   gradient fields) before spending real compute.
4. One or two full-quality real experiments previously flagged as bad
   Poisson reconstructions (`02_deflate_sphere`, `08_stratified_bands`) to
   confirm `watertight=True` on genuinely deformed geometry, not just the
   trivial translated-copy sanity check above.

(Results of each step appended below as they complete.)

## Results

1. **Syntax**: `python -m py_compile diffusion_cartogram/core.py diffusion_cartogram/__init__.py` -- clean.
2. **Inline package test** (`load_mesh_topology` + `export_mesh_file`):
   - `load_mesh_topology('skeleton_cube.stl')` -> `(69600,3)` vertices, `(139520,3)` faces, `(69600,3)` unit-length normals.
   - `method='none'` happy path (translated copy): 69600 verts / 139520 faces out, **watertight=True**.
   - `method='none'` with `original_faces=None`: raised exactly one `RuntimeWarning`, fell back to `method='poisson'` automatically; fallback result matched prior (pre-change) Poisson behavior (12068 verts / 23796 faces, watertight=False on a 5000-pt resampled cloud) -- i.e. the fallback path is a no-op relative to old behavior, as intended.
3. **Smoke test, all 9 experiments** (`--smoke`, new `load_mesh_topology`-based pipeline, `n_max=20`): all 9 ran with **zero warnings raised** (confirms `faces` is threaded through correctly for every grid-construction path, including `make_asymmetric_grid`'s custom box for experiment 06) and all 9 exported STLs came back **69600 verts / 139520 faces / watertight=True** -- exactly the source topology, as expected.
4. **Full-quality real deformations** (`--only 02,08`, `n_max=1000`, the two worst Poisson failures from the prior batch): both **watertight=True**, both still exactly 69600 verts / 139520 faces.
   - `02_deflate_sphere`: real, substantial deformation happened (per-vertex displacement mean=1.66mm, max=5.98mm) despite overall bounding-box extents landing almost back where they started (60.674 vs original 60.674) -- that's just because this particular deflate case moves the *interior* a lot (a striking pinwheel-like collapse of the interior rods toward the center, visible in the preview PNG) while the outermost corners (which define the bounding box) barely move. Not a bug; extents alone are an insensitive metric for this case.
   - `08_stratified_bands`: **notably, the previous Poisson-based run's confusing result -- internal layers appearing sheared ~45 degrees relative to each other, plus unexplained +8-10% X/Y bounding-box growth despite a Z-only density field under a fixed (Z-pinned) boundary condition -- is gone.** The no-reconstruction result instead shows the physically sensible outcome: horizontal layers shifted apart along Z (some compressed toward the bottom, some toward the top), with the vertical connecting rods visibly kinked/wavy as they stretch between displaced layers, and Z extent still nearly pinned (60.67 -> 60.72, +0.07%) as expected for a fixed boundary. This confirms the earlier sheared-diamond appearance and anomalous X/Y growth were **Poisson reconstruction artifacts** from an unevenly-redistributed point cloud, not real deformation physics -- a second, independent benefit of the no-reconstruction path beyond watertightness alone.

## Conclusion

No-reconstruction export is a clear win for this geometry: fixes watertightness on every one of the 9 experiment types (smoke-tested) and on both full-quality real reruns, and additionally eliminated a misleading reconstruction artifact that had been confounding physical interpretation of one of the 9 original results. Recommend keeping `method='none'` as the default per the plan, with `'poisson'` retained purely as the documented fallback for callers who don't have (or don't want to use) the source mesh's own topology.

## Follow-up round: subsampling, ball pivoting / alpha shape, notebook sync

### 1. Post-hoc animation point density
`animate_surface_posthoc` (`core.py`) has **no subsampling at all** -- it writes every point in `surface_points` to every frame, confirmed by `grep -n subsample core.py` returning zero matches. The visual sparsity was coming from its companion renderer, `animate_surface_deformation` (`visualization.py`), which defaulted to `subsample=5000`. Changed that default to `subsample=None` (no subsampling) and updated its docstring. `deform_utils.py`'s call site doesn't pass an explicit `subsample`, so it now renders every point automatically.

### 2. Ball pivoting + alpha shape as `export_mesh_file` methods
Added `method='ball_pivoting'` and `method='alpha_shape'` alongside `'none'`/`'poisson'`, using the same pymeshlab filters evaluated for feasibility earlier (`generate_surface_reconstruction_ball_pivoting`, `generate_alpha_shape`). New parameters, all with defaults matching pymeshlab's own except `alpha_filtering` (see below): `ball_radius=0.0` (0% = auto-estimated), `ball_clustering=20.0`, `ball_creasethr=90.0`, `ball_deletefaces=False`, `alpha=1.0`, `alpha_filtering='Alpha Shape'` (pymeshlab's own filter default is `'Alpha Complex'`, deliberately overridden -- `'Alpha Complex'` retains interior simplicial-complex faces, not just the outer boundary, per the earlier feasibility test showing 250k-330k faces vs Poisson's 65k on the same input). Docstring explicitly frames all three reconstruction methods (`poisson`/`ball_pivoting`/`alpha_shape`) as fallbacks for when `original_faces` isn't available, `method='none'` as the preferred default. Re-tested on the skeleton cube's resampled point cloud: both run without error and produce non-degenerate meshes (`ball_pivoting`: 16506 verts/12223 faces; `alpha_shape`: 28725 verts/107836 faces), consistent with the earlier feasibility finding that neither is watertight on this thin-wireframe geometry -- expected, and exactly why they're documented as fallbacks rather than the default.

### 3. Example notebook sync
Repo-wide search found exactly one notebook using mesh export: `examples/01_quickStart.ipynb`, cell `c132fc25` (`vd.export_mesh_file('deformed_cube.stl', deformed_surface)`). Its `deformed_surface` comes from `create_pcd`'s resampled point cloud (no `original_faces` available), so under the new default this would silently fall back to Poisson with a warning on every run. Updated the call to explicitly pass `method='poisson'` (identical behavior, no warning, explicit intent) with a comment pointing to `load_mesh_topology` + `method='none'` as the alternative. No other notebook references `export_mesh_file` or `export_mesh_vtk`. (Also found the same pattern in `README.md`/`README_pypi.md`'s quick-start snippet -- left as-is since the instruction scope was example notebooks specifically, flagged here for awareness.)

### Regression check
Ran the full existing test suite (`pytest tests/`) after all changes: **86 passed, 3 skipped**, only the two expected fallback `RuntimeWarning`s (from tests that call `export_mesh_file` without `original_faces`, same as the notebook case above) -- no behavioral regressions.

## Full 9-experiment run, all fixes combined

Ran the complete batch (`run_deformations.py`, full quality: `max_points=32_000`,
`n_max=1000`). Mid-run, discovered items 7 and 9 (which had already been re-run
once this session with `span_fraction=0.3`) still showed no net bounding-box
change despite genuine internal displacement (mean 0.88mm, max 4.35mm, purely
along x for item 7) -- because a `span_fraction < 1.0` holds density **flat at
the object's own edges** (zero gradient there), so real internal material moves
but the end-faces that define the bounding box never do. Fixed by changing
`span_fraction`'s default to `1.0` (ramp spans the object's *entire* own width,
so its own edges sit at the ends of the ramp where the gradient is nonzero) and
re-ran just 07 and 09 with the corrected code (items 1-6, 8 unaffected by this
issue, kept from the main run).

Final results, all **watertight=True**, all exactly 69,600 verts / 139,520
faces (topology preserved end to end):

| # | Extents (mm) vs. original 60.67 | Notes |
|---|---|---|
| 01 inflate_sphere | +16.8% all axes | Visibly rounded into a near-sphere |
| 02 deflate_sphere | ~0% (bbox), but real interior collapse | Striking pinwheel-like collapse toward center; outer corners barely move (see prior turn's per-vertex displacement check: mean 1.66mm, max 5.98mm) |
| 03 bc_free | +0.6-1.3% | Visible asymmetric corner bulge |
| 04 bc_fixed | ~0% | Stays flush/cubic, correctly contrasting with 03 |
| 05 bc_mixed_symmetric | +0.8-1.9% | Free in X/Y, flush top/bottom in Z |
| 06 bc_mixed_asymmetric | +0.5-0.8% | Bulge only at the free corner, sharp at the fixed (+X,+Y) corner |
| 07 stretch_uniform | +3.0% X only, 0% Y/Z | Correct anisotropic stretch after the span_fraction fix |
| 08 stratified_bands | ~0% (X/Y), +0.1% (Z) | Layers visibly shifted/kinked in Z; the previous run's 45-degree-sheared-layers artifact is confirmed gone |
| 09 diagonal_twist | +1.0% all axes | Diagonal density gradient visible in internal grid-line spacing |

All previews, STLs, and GIFs are in `applications/deformations/<name>/`.

## Status: recommend committing

Given: (a) the no-reconstruction default is watertight on all 9 real
deformations plus the earlier translated-copy sanity check, (b) it fixed a
genuine visual artifact (item 8's sheared layers) beyond just watertightness,
(c) full backward compatibility confirmed via the existing test suite (86
passed, 3 skipped, only the intentional fallback warnings), and (d) ball
pivoting/alpha shape are additive, isolated new code paths that don't touch
the default behavior -- this no longer looks like an experiment that "didn't
pan out." Still uncommitted per the original instruction; ready whenever you
want to commit.
