"""
Visualization tools for 2-D VDERM exports.

Mirrors visualization.py for the 2-D pipeline:

  plot_map_2d          — static scatter plot of 2-D map points
  plot_density_field_2d — heatmap of the grid density field
  animate_map_deformation_2d — GIF / MP4 from exported map CSVs
  animate_grid_deformation_2d — GIF / MP4 from exported grid CSVs
  plot_density_evolution_2d — density statistics over iterations

All animation functions read the CSV files written by
run_VDERM_2d_with_tracking().
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import glob
import os
from tqdm import tqdm

from .core_2d import read_csv_2d


# ─── Static plots ─────────────────────────────────────────────────────────────

def plot_map_2d(positions, densities=None, title='Map Points',
                cmap='plasma', figsize=(8, 7), point_size=2, alpha=0.7,
                save_file=None):
    """
    Plot a 2-D point set with optional density colour coding.

    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        [x, y] coordinates.
    densities : ndarray, shape (n,), optional
        Per-point density values.  If None, points are drawn in a
        uniform colour.
    title : str, default='Map Points'
    cmap : str, default='plasma'
    figsize : tuple, default=(8, 7)
    point_size : float, default=2
    alpha : float, default=0.7
    save_file : str, optional
        Path to save the figure.  If None the figure is shown interactively.

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    >>> pts, crs = vd.read_geojson('countries.geojson')
    >>> vd.plot_map_2d(pts, title='World Countries')

    >>> # After deformation
    >>> deformed = vd.interpolate_to_map_2d(pts, gp, grid.get_displacement_field())
    >>> dens = vd.interpolate_densities_2d(pts, grid)
    >>> vd.plot_map_2d(deformed, densities=dens, title='Cartogram')
    """
    fig, ax = plt.subplots(figsize=figsize)

    if densities is not None:
        sc = ax.scatter(positions[:, 0], positions[:, 1],
                        c=densities, cmap=cmap, s=point_size, alpha=alpha)
        plt.colorbar(sc, ax=ax, label='Density')
    else:
        ax.scatter(positions[:, 0], positions[:, 1],
                   c='dodgerblue', s=point_size, alpha=alpha)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_file}")
    else:
        plt.show()

    return fig


def plot_density_field_2d(grid, title='Density Field',
                          cmap='plasma', figsize=(8, 7), save_file=None):
    """
    Render the current grid density field as a 2-D heatmap.

    Parameters
    ----------
    grid : VDERMGrid2D
    title : str, default='Density Field'
    cmap : str, default='plasma'
    figsize : tuple, default=(8, 7)
    save_file : str, optional

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    >>> grid.set_density(lambda x, y: 1 + np.exp(-((x-0)**2+(y-0)**2)))
    >>> vd.plot_density_field_2d(grid, title='Gaussian Density')
    """
    x0, y0 = grid.min_bounds
    x1 = x0 + (grid.L - 1) * grid.h
    y1 = y0 + (grid.M - 1) * grid.h

    fig, ax = plt.subplots(figsize=figsize)
    # rho shape is (L, M) = (x_index, y_index); transpose for imshow (row=y, col=x)
    im = ax.imshow(
        grid.rho.T,
        origin='lower',
        cmap=cmap,
        extent=[x0, x1, y0, y1],
        aspect='equal',
    )
    plt.colorbar(im, ax=ax, label='Density')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title, fontsize=14, fontweight='bold')

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_file}")
    else:
        plt.show()

    return fig


def plot_map_before_after(original_points, deformed_points,
                          densities=None, title='Before / After Deformation',
                          cmap='plasma', figsize=(14, 6), point_size=2,
                          alpha=0.7, save_file=None):
    """
    Side-by-side comparison of original and deformed map points.

    Parameters
    ----------
    original_points : ndarray, shape (n, 2)
    deformed_points : ndarray, shape (n, 2)
    densities : ndarray, shape (n,), optional
        Density values to colour the deformed panel.
    title : str
    cmap : str, default='plasma'
    figsize : tuple, default=(14, 6)
    point_size : float, default=2
    alpha : float, default=0.7
    save_file : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, (ax_orig, ax_deform) = plt.subplots(1, 2, figsize=figsize)

    ax_orig.scatter(original_points[:, 0], original_points[:, 1],
                    c='steelblue', s=point_size, alpha=alpha)
    ax_orig.set_title('Original', fontsize=12)
    ax_orig.set_aspect('equal')
    ax_orig.set_xlabel('X')
    ax_orig.set_ylabel('Y')
    ax_orig.grid(True, alpha=0.3)

    if densities is not None:
        sc = ax_deform.scatter(deformed_points[:, 0], deformed_points[:, 1],
                               c=densities, cmap=cmap, s=point_size, alpha=alpha)
        plt.colorbar(sc, ax=ax_deform, label='Density')
    else:
        ax_deform.scatter(deformed_points[:, 0], deformed_points[:, 1],
                          c='coral', s=point_size, alpha=alpha)

    ax_deform.set_title('Deformed (Cartogram)', fontsize=12)
    ax_deform.set_aspect('equal')
    ax_deform.set_xlabel('X')
    ax_deform.set_ylabel('Y')
    ax_deform.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_file}")
    else:
        plt.show()

    return fig


# ─── Animations ──────────────────────────────────────────────────────────────

def animate_map_deformation_2d(export_folder='vderm_2d_exports',
                                subfolder='vderm_map',
                                output_file='map_animation.gif',
                                fps=5,
                                subsample=5000,
                                cmap='plasma',
                                figsize=(8, 7),
                                alpha=0.7):
    """
    Create an animated GIF / MP4 of 2-D map point deformation.

    Reads the CSV files written by run_VDERM_2d_with_tracking() with
    ``export_map=True``.

    Parameters
    ----------
    export_folder : str, default='vderm_2d_exports'
    subfolder : str, default='vderm_map'
    output_file : str, default='map_animation.gif'
        Output path (.gif or .mp4).
    fps : int, default=5
    subsample : int or None, default=5000
        Downsample to this many points per frame.
    cmap : str, default='plasma'
    figsize : tuple, default=(8, 7)
    alpha : float, default=0.7

    Returns
    -------
    None

    Examples
    --------
    >>> vd.animate_map_deformation_2d('vderm_2d_exports',
    ...                               output_file='cartogram.gif')
    """
    pattern = os.path.join(export_folder, subfolder, 'map_iteration_*.csv')
    files = sorted(glob.glob(pattern))

    final_pattern = os.path.join(export_folder, subfolder, 'map_final_*.csv')
    final_files = glob.glob(final_pattern)
    if final_files:
        files.append(sorted(final_files)[0])

    if not files:
        raise FileNotFoundError(
            f"No map CSV files found matching {pattern}\n"
            "Run run_VDERM_2d_with_tracking with export_map=True."
        )

    print(f"Found {len(files)} frames")

    # Determine fixed subsample indices from first frame
    pos0, _ = read_csv_2d(files[0])
    if subsample and len(pos0) > subsample:
        sub_idx = np.sort(np.random.choice(len(pos0), subsample, replace=False))
        print(f"Subsampling {len(pos0)} → {subsample} points")
    else:
        sub_idx = None

    all_positions, all_densities = [], []
    for f in tqdm(files, desc="Loading"):
        pos, dens = read_csv_2d(f)
        if sub_idx is not None:
            pos = pos[sub_idx]
            dens = dens[sub_idx] if dens is not None else None
        all_positions.append(pos)
        all_densities.append(dens)

    all_pos = np.vstack(all_positions)
    pos_min = all_pos.min(axis=0)
    pos_max = all_pos.max(axis=0)

    has_density = all_densities[0] is not None
    if has_density:
        all_dens_flat = np.hstack([d for d in all_densities if d is not None])
        dens_min, dens_max = all_dens_flat.min(), all_dens_flat.max()
    else:
        dens_min, dens_max = 0, 1

    fig, ax = plt.subplots(figsize=figsize)

    # Initial scatter for colorbar
    sc_init = ax.scatter(
        all_positions[0][:, 0], all_positions[0][:, 1],
        c=(all_densities[0] if has_density else 'dodgerblue'),
        cmap=cmap if has_density else None,
        vmin=dens_min, vmax=dens_max,
        s=1, alpha=alpha
    )
    if has_density:
        plt.colorbar(sc_init, ax=ax, label='Density')

    def update(frame):
        ax.clear()
        pos = all_positions[frame]
        dens = all_densities[frame]
        if has_density:
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=dens, cmap=cmap,
                            vmin=dens_min, vmax=dens_max, s=1, alpha=alpha)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], c='dodgerblue', s=1, alpha=alpha)
        ax.set_xlim(pos_min[0], pos_max[0])
        ax.set_ylim(pos_min[1], pos_max[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

        fname = os.path.basename(files[frame])
        if 'final' in fname:
            ax.set_title('Map: Final (Converged)', fontsize=13,
                         fontweight='bold')
        else:
            iter_num = int(fname.split('_')[-1].replace('.csv', ''))
            ax.set_title(f'Map: Iteration {iter_num}', fontsize=13)
        return ax,

    print(f"Creating animation ({fps} fps)...")
    anim = FuncAnimation(fig, update, frames=len(files),
                         interval=1000 // fps, blit=False)

    _save_animation(anim, output_file, fps)
    plt.close()
    print(f"Animation saved to: {output_file}")


def animate_grid_deformation_2d(export_folder='vderm_2d_exports',
                                 subfolder='vderm_grid',
                                 output_file='grid_animation_2d.gif',
                                 fps=5,
                                 subsample=5000,
                                 cmap='plasma',
                                 figsize=(8, 7),
                                 alpha=0.5):
    """
    Animate the 2-D grid node positions coloured by density.

    Reads CSV files written by run_VDERM_2d_with_tracking() with
    ``export_grid=True`` (5-column format: x y v_x v_y rho).

    Parameters
    ----------
    export_folder : str, default='vderm_2d_exports'
    subfolder : str, default='vderm_grid'
    output_file : str, default='grid_animation_2d.gif'
    fps : int, default=5
    subsample : int or None, default=5000
    cmap : str, default='plasma'
    figsize : tuple, default=(8, 7)
    alpha : float, default=0.5

    Returns
    -------
    None
    """
    pattern = os.path.join(export_folder, subfolder, 'grid_iteration_*.csv')
    files = sorted(glob.glob(pattern))
    final_files = glob.glob(
        os.path.join(export_folder, subfolder, 'grid_final_*.csv')
    )
    if final_files:
        files.append(sorted(final_files)[0])

    if not files:
        raise FileNotFoundError(
            f"No grid CSV files found at {pattern}\n"
            "Run run_VDERM_2d_with_tracking with export_grid=True."
        )

    print(f"Found {len(files)} frames")

    # Grid CSVs have 5 columns: x y v_x v_y rho
    def _load_grid_csv(path):
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pos = data[:, :2]
        rho = data[:, 4] if data.shape[1] >= 5 else None
        return pos, rho

    pos0, _ = _load_grid_csv(files[0])
    if subsample and len(pos0) > subsample:
        sub_idx = np.sort(np.random.choice(len(pos0), subsample, replace=False))
    else:
        sub_idx = None

    all_positions, all_densities = [], []
    for f in tqdm(files, desc="Loading"):
        pos, rho = _load_grid_csv(f)
        if sub_idx is not None:
            pos = pos[sub_idx]
            rho = rho[sub_idx] if rho is not None else None
        all_positions.append(pos)
        all_densities.append(rho)

    all_pos = np.vstack(all_positions)
    pos_min = all_pos.min(axis=0)
    pos_max = all_pos.max(axis=0)

    has_density = all_densities[0] is not None
    if has_density:
        all_d = np.hstack([d for d in all_densities if d is not None])
        dens_min, dens_max = all_d.min(), all_d.max()
    else:
        dens_min, dens_max = 0, 1

    fig, ax = plt.subplots(figsize=figsize)
    sc_init = ax.scatter(
        all_positions[0][:, 0], all_positions[0][:, 1],
        c=(all_densities[0] if has_density else 'steelblue'),
        cmap=cmap if has_density else None,
        vmin=dens_min, vmax=dens_max,
        s=1, alpha=alpha
    )
    if has_density:
        plt.colorbar(sc_init, ax=ax, label='Density')

    def update(frame):
        ax.clear()
        pos = all_positions[frame]
        dens = all_densities[frame]
        if has_density:
            ax.scatter(pos[:, 0], pos[:, 1], c=dens, cmap=cmap,
                       vmin=dens_min, vmax=dens_max, s=1, alpha=alpha)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], c='steelblue', s=1, alpha=alpha)
        ax.set_xlim(pos_min[0], pos_max[0])
        ax.set_ylim(pos_min[1], pos_max[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        fname = os.path.basename(files[frame])
        if 'final' in fname:
            ax.set_title('Grid: Final (Converged)', fontsize=13,
                         fontweight='bold')
        else:
            iter_num = int(fname.split('_')[-1].replace('.csv', ''))
            ax.set_title(f'Grid: Iteration {iter_num}', fontsize=13)
        return ax,

    print(f"Creating animation ({fps} fps)...")
    anim = FuncAnimation(fig, update, frames=len(files),
                         interval=1000 // fps, blit=False)
    _save_animation(anim, output_file, fps)
    plt.close()
    print(f"Animation saved to: {output_file}")


def plot_density_evolution_2d(export_folder='vderm_2d_exports',
                               grid_folder='vderm_grid',
                               output_file='density_evolution_2d.png'):
    """
    Plot mean, min, max, and std-dev of the grid density over iterations.

    Reads CSV files written by run_VDERM_2d_with_tracking() with
    ``export_grid=True``.

    Parameters
    ----------
    export_folder : str
    grid_folder : str
    output_file : str

    Examples
    --------
    >>> vd.plot_density_evolution_2d('vderm_2d_exports')
    """
    pattern = os.path.join(export_folder, grid_folder, 'grid_iteration_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No grid CSV files at {pattern}")

    iterations, means, maxs, mins, stds = [], [], [], [], []

    for f in tqdm(files, desc="Analysing"):
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 5:
            continue
        dens = data[:, 4]
        fname = os.path.basename(f)
        iter_num = int(fname.split('_')[-1].replace('.csv', ''))
        iterations.append(iter_num)
        means.append(dens.mean())
        maxs.append(dens.max())
        mins.append(dens.min())
        stds.append(dens.std())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(iterations, means, 'b-', label='Mean', linewidth=2)
    ax1.plot(iterations, maxs, 'r--', label='Max', linewidth=1.5)
    ax1.plot(iterations, mins, 'g--', label='Min', linewidth=1.5)
    ax1.fill_between(iterations, mins, maxs, alpha=0.2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Density')
    ax1.set_title('Density Statistics (2-D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(iterations, stds, 'purple', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Std Dev')
    ax2.set_title('Density Variation Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Density evolution plot saved to: {output_file}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_animation(anim, output_file, fps):
    """Save a FuncAnimation to .gif or .mp4."""
    if output_file.endswith('.gif'):
        anim.save(output_file, writer=PillowWriter(fps=fps))
    elif output_file.endswith('.mp4'):
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(output_file, writer=FFMpegWriter(fps=fps, bitrate=1800))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save MP4 — is ffmpeg installed?\n{exc}"
            ) from exc
    else:
        raise ValueError(
            f"output_file must end with .gif or .mp4, got: {output_file}"
        )
