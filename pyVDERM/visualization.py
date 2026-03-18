"""
Visualization and animation tools for VDERM exports.

This module provides functions to create animations and visualizations from
exported VDERM data (grids, surfaces, and meshes).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
from tqdm import tqdm
import warnings

from .core import read_xyz, HAS_PYMESHLAB


def animate_grid_deformation(export_folder='vderm_exports', 
                             subfolder='vderm_grid',
                             output_file='grid_animation.gif',
                             fps=5,
                             subsample=5000,
                             cmap='plasma',
                             figsize=(15, 5),
                             alpha=0.3):
    """
    Create animated GIF of grid deformation showing three orthogonal 2D projections.
    
    Creates three side-by-side 2D plots showing XY, XZ, and YZ projections of the
    grid with density coloring. Points are overlaid with transparency to show
    the full 3D structure.
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder
    subfolder : str, default='vderm_grid'
        Subfolder containing grid exports
    output_file : str, default='grid_animation.gif'
        Output animation filename (.gif or .mp4)
    fps : int, default=5
        Frames per second
    subsample : int or None, default=5000
        Subsample to this many points for faster rendering.
        If None, use all points (may be slow for large grids).
    cmap : str, default='plasma'
        Matplotlib colormap name for density visualization
    figsize : tuple, default=(15, 5)
        Figure size in inches (width, height)
    alpha : float, default=0.3
        Opacity for overlaid points (0=transparent, 1=opaque)
    
    Returns
    -------
    None
        Saves animation to output_file
    
    Examples
    --------
    >>> # Basic usage
    >>> animate_grid_deformation('my_exports', output_file='deform.gif')
    
    >>> # More opaque points
    >>> animate_grid_deformation('my_exports', alpha=0.5)
    
    >>> # Higher frame rate
    >>> animate_grid_deformation('my_exports', fps=10, subsample=3000)
    """
    
    # Find all grid files
    pattern = os.path.join(export_folder, subfolder, 'grid_iteration_*.xyz')
    files = sorted(glob.glob(pattern))
    
    # Also check for final file
    final_pattern = os.path.join(export_folder, subfolder, 'grid_final_*.xyz')
    final_files = glob.glob(final_pattern)
    if final_files:
        files.append(final_files[0])
    
    if len(files) == 0:
        raise FileNotFoundError(
            f"No grid files found matching {pattern}\n"
            f"Make sure you ran run_VDERM_with_tracking with export_grid=True"
        )
    
    print(f"Found {len(files)} frames to animate")
    
    # Load first frame to determine subsample indices
    pos_first, norm_first, dens_first = read_xyz(files[0])
    
    # Determine subsample indices
    if subsample and len(pos_first) > subsample:
        subsample_indices = np.random.choice(len(pos_first), subsample, replace=False)
        subsample_indices.sort()  # Sort for better cache locality
        print(f"Subsampling {len(pos_first)} points to {subsample} points")
    else:
        subsample_indices = None
        print(f"Using all {len(pos_first)} points (no subsampling)")
    
    # Load all data
    all_positions = []
    all_densities = []
    
    for f in tqdm(files, desc="Loading data"):
        pos, norm, dens = read_xyz(f)
        
        # Subsample if requested
        if subsample_indices is not None:
            pos = pos[subsample_indices]
            dens = dens[subsample_indices]
        
        all_positions.append(pos)
        all_densities.append(dens)
    
    # Get global bounds for consistent axes
    all_pos = np.vstack(all_positions)
    pos_min = all_pos.min(axis=0)
    pos_max = all_pos.max(axis=0)
    
    all_dens = np.hstack(all_densities)
    dens_min = all_dens.min()
    dens_max = all_dens.max()
    
    print(f"Position range: [{pos_min[0]:.3f}, {pos_max[0]:.3f}] × "
          f"[{pos_min[1]:.3f}, {pos_max[1]:.3f}] × "
          f"[{pos_min[2]:.3f}, {pos_max[2]:.3f}]")
    print(f"Density range: [{dens_min:.3f}, {dens_max:.3f}]")
    
    # Create figure with three 2D subplots
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=figsize)
    
    # Create colormap normalizer
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=dens_min, vmax=dens_max)
    
    def update(frame):
        """Update function for animation"""
        # Clear all axes
        ax_xy.clear()
        ax_xz.clear()
        ax_yz.clear()
        
        positions = all_positions[frame]
        densities = all_densities[frame]
        
        # XY plane (top view - looking down Z axis)
        ax_xy.scatter(positions[:, 0], positions[:, 1], 
                     c=densities, cmap=cmap, s=1, alpha=alpha,
                     vmin=dens_min, vmax=dens_max)
        ax_xy.set_xlim(pos_min[0], pos_max[0])
        ax_xy.set_ylim(pos_min[1], pos_max[1])
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.set_title('XY Plane (Top View)')
        ax_xy.set_aspect('equal')
        ax_xy.grid(True, alpha=0.3)
        
        # XZ plane (front view - looking along Y axis)
        ax_xz.scatter(positions[:, 0], positions[:, 2],
                     c=densities, cmap=cmap, s=1, alpha=alpha,
                     vmin=dens_min, vmax=dens_max)
        ax_xz.set_xlim(pos_min[0], pos_max[0])
        ax_xz.set_ylim(pos_min[2], pos_max[2])
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title('XZ Plane (Front View)')
        ax_xz.set_aspect('equal')
        ax_xz.grid(True, alpha=0.3)
        
        # YZ plane (side view - looking along X axis)
        scatter_yz = ax_yz.scatter(positions[:, 1], positions[:, 2],
                                   c=densities, cmap=cmap, s=1, alpha=alpha,
                                   vmin=dens_min, vmax=dens_max)
        ax_yz.set_xlim(pos_min[1], pos_max[1])
        ax_yz.set_ylim(pos_min[2], pos_max[2])
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.set_title('YZ Plane (Side View)')
        ax_yz.set_aspect('equal')
        ax_yz.grid(True, alpha=0.3)
        
        # Extract iteration number and set main title
        filename = os.path.basename(files[frame])
        if 'final' in filename:
            fig.suptitle('Final Grid State (Converged)', 
                        fontsize=16, fontweight='bold')
        else:
            iter_num = int(filename.split('_')[-1].replace('.xyz', ''))
            fig.suptitle(f'Grid Deformation - Iteration {iter_num}', 
                        fontsize=16)
        
        return scatter_yz,  # Return one for blit
    
    # Create initial frame to set up colorbar
    positions = all_positions[0]
    densities = all_densities[0]
    
    scatter_yz = ax_yz.scatter(positions[:, 1], positions[:, 2],
                              c=densities, cmap=cmap, s=1, alpha=alpha,
                              vmin=dens_min, vmax=dens_max)
    
    # Add colorbar (shared for all three plots)
    cbar = fig.colorbar(scatter_yz, ax=[ax_xy, ax_xz, ax_yz], 
                       label='Density', shrink=0.8, pad=0.02)
    
    # Create animation
    print(f"Creating animation with {fps} fps...")
    anim = FuncAnimation(fig, update, frames=len(files), 
                        interval=1000//fps, blit=False)
    
    # Save
    print(f"Saving to {output_file}...")
    if output_file.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
    elif output_file.endswith('.mp4'):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_file, writer=writer)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save MP4. Make sure ffmpeg is installed.\n"
                f"Error: {e}\n"
                f"Try saving as .gif instead, or install ffmpeg."
            )
    else:
        raise ValueError(
            f"output_file must end with .gif or .mp4, got: {output_file}"
        )
    
    plt.close()
    print(f"✓ Animation saved to: {output_file}")


def animate_surface_deformation(export_folder='vderm_exports',
                                subfolder='vderm_surface', 
                                output_file='surface_animation.gif',
                                fps=5,
                                subsample=5000,
                                show_normals=False,
                                alpha=0.6,
                                figsize=(10, 8)):
    """
    Create animated GIF of surface deformation from exported surface states.
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder
    subfolder : str, default='vderm_surface'
        Subfolder containing surface exports
    output_file : str, default='surface_animation.gif'
        Output animation filename (.gif or .mp4)
    fps : int, default=5
        Frames per second
    subsample : int or None, default=5000
        Subsample to this many points for faster rendering
    show_normals : bool, default=False
        If True, draw normal vectors (slower, may clutter visualization)
    figsize : tuple, default=(10, 8)
        Figure size in inches
    
    Returns
    -------
    None
        Saves animation to output_file
    
    Examples
    --------
    >>> # Basic surface animation
    >>> animate_surface_deformation('my_exports')
    
    >>> # Show normal vectors
    >>> animate_surface_deformation('my_exports',
    ...                            show_normals=True,
    ...                            subsample=1000)  # Use fewer points with normals
    """
    
    # Find all surface files
    pattern = os.path.join(export_folder, subfolder, 'surface_iteration_*.xyz')
    files = sorted(glob.glob(pattern))
    
    # Check for final
    final_pattern = os.path.join(export_folder, subfolder, 'surface_final_*.xyz')
    final_files = glob.glob(final_pattern)
    if final_files:
        files.append(final_files[0])
    
    if len(files) == 0:
        raise FileNotFoundError(
            f"No surface files found matching {pattern}\n"
            f"Make sure you ran run_VDERM_with_tracking with export_surface=True"
        )
    
    print(f"Found {len(files)} frames to animate")
    
    # Load first frame to determine subsample indices
    pos_first, norm_first, dense_first = read_xyz(files[0])

    min_dens = min(dense_first)
    max_dens = max(dense_first)
    
    # Determine subsample indices 
    if subsample and len(pos_first) > subsample:
        subsample_indices = np.random.choice(len(pos_first), subsample, replace=False)
        subsample_indices.sort()  
        print(f"Subsampling {len(pos_first)} points to {subsample} points")
    else:
        subsample_indices = None
        print(f"Using all {len(pos_first)} points (no subsampling)")
    
    # Load all data
    all_points = []
    all_normals = []
    all_dens = []
    
    for f in tqdm(files, desc="Loading data"):
        pts, norms, dens = read_xyz(f)
        
        # Subsample if requested
        if subsample_indices is not None:
            pts = pts[subsample_indices]
            norms = norms[subsample_indices]
            dens = dens[subsample_indices]
        
        all_points.append(pts)
        all_normals.append(norms)
        all_dens.append(dens)
    
    # Get global bounds
    all_pts = np.vstack(all_points)
    pts_min = all_pts.min(axis=0)
    pts_max = all_pts.max(axis=0)
    
    print(f"Surface bounds: [{pts_min[0]:.3f}, {pts_max[0]:.3f}] × "
          f"[{pts_min[1]:.3f}, {pts_max[1]:.3f}] × "
          f"[{pts_min[2]:.3f}, {pts_max[2]:.3f}]")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        
        points = all_points[frame]
        normals = all_normals[frame]
        densities = all_dens[frame]
        
        # Plot surface
        box = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=densities,cmap='viridis', s=1, alpha=alpha,
                  vmin=min_dens,vmax=max_dens)
        
        # Optionally show normals
        if show_normals:
            # Only show every Nth normal for clarity
            step = max(1, len(points) // 100)
            normal_scale = (pts_max - pts_min).mean() * 0.02  # Scale normals to 2% of domain
            ax.quiver(points[::step, 0], points[::step, 1], points[::step, 2],
                     normals[::step, 0], normals[::step, 1], normals[::step, 2],
                     length=normal_scale, color='red', alpha=0.5, 
                     arrow_length_ratio=0.3)
        
        ax.set_xlim(pts_min[0], pts_max[0])
        ax.set_ylim(pts_min[1], pts_max[1])
        ax.set_zlim(pts_min[2], pts_max[2])
        
        ax.set_box_aspect([
        pts_max[0] - pts_min[0],
        pts_max[1] - pts_min[1],
        pts_max[2] - pts_min[2]
        ])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Extract iteration
        filename = os.path.basename(files[frame])
        if 'final' in filename:
            ax.set_title('Surface: Final (Converged)', fontsize=14, fontweight='bold')
        else:
            iter_num = int(filename.split('_')[-1].replace('.xyz', ''))
            ax.set_title(f'Surface: Iteration {iter_num}', fontsize=14)
        
        return ax,

    # initial frame to set up colorbar

    box = ax.scatter(pos_first[:, 0], pos_first[:, 1], pos_first[:, 2],
                  c=dense_first,cmap='viridis', s=1, alpha=alpha,
                  vmin=min_dens,vmax=max_dens)

    cbar = plt.colorbar(box, ax=ax, label='Density', shrink=0.5)
    
    # Create animation
    print(f"Creating animation with {fps} fps...")
    anim = FuncAnimation(fig, update, frames=len(files),
                        interval=1000//fps, blit=False)
    
    # Save
    print(f"Saving to {output_file}...")
    if output_file.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
    elif output_file.endswith('.mp4'):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_file, writer=writer)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save MP4. Make sure ffmpeg is installed.\n"
                f"Error: {e}"
            )
    else:
        raise ValueError("output_file must end with .gif or .mp4")
    
    plt.close()
    print(f"✓ Animation saved to: {output_file}")
    

def interactive_pcd_plot(export_folder, pattern='grid_iteration_*.xyz', 
                            subsample=True, max_points=10_000):
    """
    Visualize a sequence of exported grid states to see deformation over time. 
    
    Parameters
    ----------
    export_folder : str
        Folder containing exported grid files
    pattern : str
        Glob pattern to match grid files (default: 'grid_iteration_*.xyz')
    subsample : bool
        Whether to downsample the number of points plotted
    max_points : 
        max points to plot if subsampling enabled
    
    Examples
    --------
    >>> visualize_grid_sequence('deformation_sequence')
    >>> visualize_grid_sequence('exports', save_animation=True, output_file='deform.gif')

    Notes
    -----
    ipympl and ipywidgets must be installed, and the jupyter backend must be set 
    to %matplotlib widgets for this function to work. Otherwise it will produce a 
    static plot of the first frame
    """
    
    # Find all matching files
    file_pattern = os.path.join(export_folder, pattern)
    files = sorted(glob.glob(file_pattern))
    
    if len(files) == 0:
        raise FileNotFoundError(f"No files matching pattern '{file_pattern}'")
    
    print(f"Found {len(files)} files to visualize")
    
    # Read all files to get global bounds and density range
    all_positions = []
    all_densities = []
    for f in files:
        pos, norms, dens = read_xyz(f)
        all_positions.append(pos)
        all_densities.append(dens)
    
    # Compute global bounds for consistent axes
    all_pos_concat = np.vstack(all_positions)
    all_dens_concat = np.hstack(all_densities)
    
    pos_min = all_pos_concat.min(axis=0)
    pos_max = all_pos_concat.max(axis=0)
    dens_min = all_dens_concat.min()
    dens_max = all_dens_concat.max()
    
    # Interactive visualization
    from matplotlib.widgets import Slider
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial plot
    positions, densities = all_positions[0], all_densities[0]
    
    # Subsample if needed
    if subsample and len(positions) > max_points:
        indices = np.random.choice(len(positions), 10000, replace=False)
        positions_sub = positions[indices]
        densities_sub = densities[indices]
    else:
        positions_sub = positions
        densities_sub = densities
    
    scatter = ax.scatter(positions_sub[:, 0], positions_sub[:, 1], positions_sub[:, 2],
                        c=densities_sub, cmap='viridis', s=1,
                        vmin=dens_min, vmax=dens_max)
    
    ax.set_xlim(pos_min[0], pos_max[0])
    ax.set_ylim(pos_min[1], pos_max[1])
    ax.set_zlim(pos_min[2], pos_max[2])
    
    ax.set_box_aspect([
    pos_max[0] - pos_min[0],
    pos_max[1] - pos_min[1],
    pos_max[2] - pos_min[2]
    ])
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Iteration 0')
    
    cbar = plt.colorbar(scatter, ax=ax, label='Density', shrink=0.5)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Iteration', 0, len(files)-1, 
                    valinit=0, valstep=1)
    
    def update_plot(val):
        frame = int(slider.val)
        positions, densities = all_positions[frame], all_densities[frame]
        
        # Subsample if needed
        if subsample and len(positions) > max_points:
            positions_sub = positions[indices]
            densities_sub = densities[indices]
        else:
            positions_sub = positions
            densities_sub = densities
        
        scatter._offsets3d = (positions_sub[:, 0], 
                                positions_sub[:, 1], 
                                positions_sub[:, 2])
        scatter.set_array(densities_sub)
        
        ax.set_title(f'Iteration {frame * 10}')  # Adjust based on export_frequency
        fig.canvas.draw_idle()
    
    slider.on_changed(update_plot)
    plt.show()


def create_side_by_side_animation(export_folder='vderm_exports',
                                   grid_folder='vderm_grid',
                                   surface_folder='vderm_surface',
                                   output_file='comparison.gif',
                                   fps=5,
                                   subsample=3000,
                                   alpha_grid=1,
                                   alpha_surface=0.6,
                                   figsize=(15, 10)):
    """
    Create side-by-side animation showing grid and surface evolution in 2D projections.
    
    Creates a 2×3 grid of subplots:
    - Top row: Grid XY, XZ, YZ projections (with density coloring)
    - Bottom row: Surface XY, XZ, YZ projections (solid color)
    
    Grid and surface use their own separate bounds for better visualization.
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder
    grid_folder : str, default='vderm_grid'
        Subfolder containing grid exports
    surface_folder : str, default='vderm_surface'
        Subfolder containing surface exports
    output_file : str, default='comparison.gif'
        Output animation filename (.gif or .mp4)
    fps : int, default=5
        Frames per second
    subsample : int or None, default=3000
        Subsample each dataset to this many points
    alpha : float, default=0.3
        Opacity for grid points
    figsize : tuple, default=(15, 10)
        Figure size in inches (width, height)
    
    Returns
    -------
    None
        Saves animation to output_file
    
    Notes
    -----
    Grid and surface exports must have matching iteration numbers.
    If they don't match, only overlapping iterations will be animated.
    
    Examples
    --------
    >>> create_side_by_side_animation('my_exports', output_file='both.gif')
    
    >>> # More opaque grid, faster playback
    >>> create_side_by_side_animation('my_exports', alpha=0.5, fps=10)
    """
    
    # Find files
    grid_pattern = os.path.join(export_folder, grid_folder, 'grid_iteration_*.xyz')
    surface_pattern = os.path.join(export_folder, surface_folder, 'surface_iteration_*.xyz')
    
    grid_files = sorted(glob.glob(grid_pattern))
    surface_files = sorted(glob.glob(surface_pattern))
    
    if len(grid_files) == 0:
        raise FileNotFoundError(f"No grid files found at {grid_pattern}")
    if len(surface_files) == 0:
        raise FileNotFoundError(f"No surface files found at {surface_pattern}")
    
    # Must have matching counts
    if len(grid_files) != len(surface_files):
        print(f"Warning: Found {len(grid_files)} grid files but "
              f"{len(surface_files)} surface files")
        n_frames = min(len(grid_files), len(surface_files))
        grid_files = grid_files[:n_frames]
        surface_files = surface_files[:n_frames]
    
    print(f"Creating {len(grid_files)} frame side-by-side animation")
    
    # Load first frame to determine subsample indices
    pos_first, norm_first, dense_first = read_xyz(surface_files[0])
    
    # Determine subsample indices ONCE (use for all frames)
    if subsample and len(pos_first) > subsample:
        subsample_surface_indices = np.random.choice(len(pos_first), subsample, replace=False)
        subsample_surface_indices.sort()  # Sort for better cache locality
        print(f"Subsampling {len(pos_first)} points to {subsample} points")
    else:
        subsample_surface_indices = None
        print(f"Using all {len(pos_first)} points (no subsampling)")
        
    # Load first frame to determine subsample indices
    pos_first, norm_first, dense_first = read_xyz(grid_files[0])
    
    # Determine subsample indices ONCE (use for all frames)
    if subsample and len(pos_first) > subsample:
        subsample_grid_indices = np.random.choice(len(pos_first), subsample, replace=False)
        subsample_grid_indices.sort()  # Sort for better cache locality
        print(f"Subsampling {len(pos_first)} points to {subsample} points")
    else:
        subsample_grid_indices = None
        print(f"Using all {len(pos_first)} points (no subsampling)")
    
    # Load all data
    all_grid_pos, all_grid_dens = [], []
    all_surf_pts, all_surf_norms = [], []
    
    for gf, sf in tqdm(zip(grid_files, surface_files), 
                       total=len(grid_files), desc="Loading"):
        gp, gn, gd = read_xyz(gf)
        sp, sn, sd = read_xyz(sf)
        
        # Subsample
        if subsample:
            gp, gd = gp[subsample_grid_indices], gd[subsample_grid_indices]
            sp, sn = sp[subsample_surface_indices], sn[subsample_surface_indices]
        
        all_grid_pos.append(gp)
        all_grid_dens.append(gd)
        all_surf_pts.append(sp)
        all_surf_norms.append(sn)
    
    # Get separate bounds for grid and surface
    all_grid = np.vstack(all_grid_pos)
    grid_min = all_grid.min(axis=0)
    grid_max = all_grid.max(axis=0)
    
    all_surf = np.vstack(all_surf_pts)
    surf_min = all_surf.min(axis=0)
    surf_max = all_surf.max(axis=0)
    
    all_dens = np.hstack(all_grid_dens)
    dens_min, dens_max = all_dens.min(), all_dens.max()
    
    print(f"Grid bounds: [{grid_min[0]:.3f}, {grid_max[0]:.3f}] × "
          f"[{grid_min[1]:.3f}, {grid_max[1]:.3f}] × "
          f"[{grid_min[2]:.3f}, {grid_max[2]:.3f}]")
    print(f"Surface bounds: [{surf_min[0]:.3f}, {surf_max[0]:.3f}] × "
          f"[{surf_min[1]:.3f}, {surf_max[1]:.3f}] × "
          f"[{surf_min[2]:.3f}, {surf_max[2]:.3f}]")
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax_g_xy, ax_g_xz, ax_g_yz = axes[0]  # Top row: grid
    ax_s_xy, ax_s_xz, ax_s_yz = axes[1]  # Bottom row: surface
    
    def update(frame):
        """Update function for animation"""
        # Clear all axes
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        grid_pos = all_grid_pos[frame]
        grid_dens = all_grid_dens[frame]
        surf_pts = all_surf_pts[frame]
        
        # ==========================================
        # Top row: Grid projections (with density)
        # ==========================================
        
        # Grid XY
        ax_g_xy.scatter(grid_pos[:, 0], grid_pos[:, 1],
                       c=grid_dens, cmap='plasma', s=1, alpha=alpha_grid,
                       vmin=dens_min, vmax=dens_max)
        ax_g_xy.set_xlim(grid_min[0], grid_max[0])
        ax_g_xy.set_ylim(grid_min[1], grid_max[1])
        ax_g_xy.set_xlabel('X')
        ax_g_xy.set_ylabel('Y')
        ax_g_xy.set_title('Grid XY (Top View)', fontsize=10)
        ax_g_xy.set_aspect('equal')
        ax_g_xy.grid(True, alpha=0.2)
        
        # Grid XZ
        ax_g_xz.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=grid_dens, cmap='plasma', s=1, alpha=alpha_grid,
                       vmin=dens_min, vmax=dens_max)
        ax_g_xz.set_xlim(grid_min[0], grid_max[0])
        ax_g_xz.set_ylim(grid_min[2], grid_max[2])
        ax_g_xz.set_xlabel('X')
        ax_g_xz.set_ylabel('Z')
        ax_g_xz.set_title('Grid XZ (Front View)', fontsize=10)
        ax_g_xz.set_aspect('equal')
        ax_g_xz.grid(True, alpha=0.2)
        
        # Grid YZ
        scatter_grid = ax_g_yz.scatter(grid_pos[:, 1], grid_pos[:, 2],
                                       c=grid_dens, cmap='plasma', s=1, alpha=alpha_grid,
                                       vmin=dens_min, vmax=dens_max)
        ax_g_yz.set_xlim(grid_min[1], grid_max[1])
        ax_g_yz.set_ylim(grid_min[2], grid_max[2])
        ax_g_yz.set_xlabel('Y')
        ax_g_yz.set_ylabel('Z')
        ax_g_yz.set_title('Grid YZ (Side View)', fontsize=10)
        ax_g_yz.set_aspect('equal')
        ax_g_yz.grid(True, alpha=0.2)
        
        # ==========================================
        # Bottom row: Surface projections
        # ==========================================
        
        # Surface XY
        ax_s_xy.scatter(surf_pts[:, 0], surf_pts[:, 1],
                       c='dodgerblue', s=1, alpha=alpha_surface)
        ax_s_xy.set_xlim(surf_min[0], surf_max[0])
        ax_s_xy.set_ylim(surf_min[1], surf_max[1])
        ax_s_xy.set_xlabel('X')
        ax_s_xy.set_ylabel('Y')
        ax_s_xy.set_title('Surface XY (Top View)', fontsize=10)
        ax_s_xy.set_aspect('equal')
        ax_s_xy.grid(True, alpha=0.2)
        
        # Surface XZ
        ax_s_xz.scatter(surf_pts[:, 0], surf_pts[:, 2],
                       c='dodgerblue', s=1, alpha=alpha_surface)
        ax_s_xz.set_xlim(surf_min[0], surf_max[0])
        ax_s_xz.set_ylim(surf_min[2], surf_max[2])
        ax_s_xz.set_xlabel('X')
        ax_s_xz.set_ylabel('Z')
        ax_s_xz.set_title('Surface XZ (Front View)', fontsize=10)
        ax_s_xz.set_aspect('equal')
        ax_s_xz.grid(True, alpha=0.2)
        
        # Surface YZ
        ax_s_yz.scatter(surf_pts[:, 1], surf_pts[:, 2],
                       c='dodgerblue', s=1, alpha=alpha_surface)
        ax_s_yz.set_xlim(surf_min[1], surf_max[1])
        ax_s_yz.set_ylim(surf_min[2], surf_max[2])
        ax_s_yz.set_xlabel('Y')
        ax_s_yz.set_ylabel('Z')
        ax_s_yz.set_title('Surface YZ (Side View)', fontsize=10)
        ax_s_yz.set_aspect('equal')
        ax_s_yz.grid(True, alpha=0.2)
        
        # Extract iteration and set main title
        filename = os.path.basename(grid_files[frame])
        if 'final' in filename:
            fig.suptitle('Grid vs Surface: Final (Converged)', 
                        fontsize=14, fontweight='bold')
        else:
            iter_num = int(filename.split('_')[-1].replace('.xyz', ''))
            fig.suptitle(f'Grid vs Surface: Iteration {iter_num}', 
                        fontsize=14)
        
        return scatter_grid,
    
    # Create initial frame for colorbar
    grid_pos = all_grid_pos[0]
    grid_dens = all_grid_dens[0]
    scatter_grid = ax_g_yz.scatter(grid_pos[:, 1], grid_pos[:, 2],
                                   c=grid_dens, cmap='plasma', s=1, alpha=alpha_grid,
                                   vmin=dens_min, vmax=dens_max)
    
    # Add colorbar for grid density (spans top row)
    cbar = fig.colorbar(scatter_grid, ax=axes[0, :],
                       label='Grid Density', shrink=0.8, pad=0.02,
                       orientation='vertical', location='right')
    
    print(f"Creating animation with {fps} fps...")
    anim = FuncAnimation(fig, update, frames=len(grid_files),
                        interval=1000//fps, blit=False)
    
    print(f"Saving to {output_file}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    plt.close()
    print(f"✓ Animation saved to: {output_file}")


def plot_density_evolution(export_folder='vderm_exports',
                           grid_folder='vderm_grid',
                           output_file='density_evolution.png'):
    """
    Plot how mean and max density evolve over iterations.
    
    Parameters
    ----------
    export_folder : str
        Base export folder
    grid_folder : str
        Subfolder containing grid exports
    output_file : str
        Output plot filename
    
    Examples
    --------
    >>> plot_density_evolution('my_exports')
    """
    
    pattern = os.path.join(export_folder, grid_folder, 'grid_iteration_*.xyz')
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise FileNotFoundError(f"No grid files found at {pattern}")
    
    iterations = []
    mean_densities = []
    max_densities = []
    min_densities = []
    std_densities = []
    
    for f in tqdm(files, desc="Analyzing density"):
        pos, norm, densities = read_xyz(f)
        
        # Extract iteration number
        filename = os.path.basename(f)
        iter_num = int(filename.split('_')[-1].replace('.xyz', ''))
        
        iterations.append(iter_num)
        mean_densities.append(densities.mean())
        max_densities.append(densities.max())
        min_densities.append(densities.min())
        std_densities.append(densities.std())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Mean, min, max
    ax1.plot(iterations, mean_densities, 'b-', label='Mean', linewidth=2)
    ax1.plot(iterations, max_densities, 'r--', label='Max', linewidth=1.5)
    ax1.plot(iterations, min_densities, 'g--', label='Min', linewidth=1.5)
    ax1.fill_between(iterations, min_densities, max_densities, alpha=0.2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Density')
    ax1.set_title('Density Statistics Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Standard deviation
    ax2.plot(iterations, std_densities, 'purple', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Density Std Dev')
    ax2.set_title('Density Variation Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Density evolution plot saved to: {output_file}")

def export_grid_to_paraview(export_folder='vderm_exports',
                            subfolder='vderm_grid',
                            output_folder='paraview_grid'):
    """
    Export grid data to VTK format for visualization in ParaView.
    
    Exports grid node positions with all available attributes:
    velocity vectors and density values (if present).
    
    Grid files are expected to have format: x y z v_x v_y v_z rho
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder
    subfolder : str, default='vderm_grid'
        Subfolder containing grid exports
    output_folder : str, default='paraview_grid'
        Folder to save VTK files
    
    Returns
    -------
    None
        VTK files saved to output_folder
    
    Examples
    --------
    >>> export_grid_to_paraview('my_exports')
    >>> # Then in ParaView: File → Open → select .vtk files
    >>> # Color by 'density' to see density field evolution
    >>> # Apply 'Glyph' filter with 'velocity' to visualize flow field
    """
    
    pattern = os.path.join(export_folder, subfolder, '*.xyz')
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise FileNotFoundError(
            f"No grid files found at {pattern}\n"
            f"Make sure you ran run_VDERM_with_tracking with export_grid=True"
        )
    
    os.makedirs(output_folder, exist_ok=True)
    
    for f in tqdm(files, desc="Converting grid to VTK"):
        # Read grid data (should have positions + velocities + densities)
        positions, velocities, densities = read_xyz(f)
        
        # Check what data is available
        has_velocities = velocities is not None
        has_densities = densities is not None
        
        if not has_velocities:
            warnings.warn(
                f"File {f} does not contain velocity vectors. "
                "Grid exports should have 7 columns (x y z v_x v_y v_z rho)."
            )
        
        if not has_densities:
            warnings.warn(
                f"File {f} does not contain density data. "
                "Grid exports should have 7 columns (x y z v_x v_y v_z rho)."
            )
        
        # Create output filename
        basename = os.path.basename(f).replace('.xyz', '.vtk')
        output_path = os.path.join(output_folder, basename)
        
        # Write VTK file with all available attributes
        with open(output_path, 'w') as vtk:
            vtk.write('# vtk DataFile Version 3.0\n')
            vtk.write('VDERM Grid Data\n')
            vtk.write('ASCII\n')
            vtk.write('DATASET POLYDATA\n')
            
            # Write vertices
            vtk.write(f'POINTS {len(positions)} float\n')
            for pos in positions:
                vtk.write(f'{pos[0]:.6e} {pos[1]:.6e} {pos[2]:.6e}\n')
            
            # Write point data (only if we have velocities or densities)
            if has_velocities or has_densities:
                vtk.write(f'\nPOINT_DATA {len(positions)}\n')
                
                # Add velocities as vector data
                if has_velocities:
                    vtk.write('VECTORS velocity float\n')
                    for vel in velocities:
                        vtk.write(f'{vel[0]:.6e} {vel[1]:.6e} {vel[2]:.6e}\n')
                
                # Add density as scalar data
                if has_densities:
                    vtk.write('\nSCALARS density float 1\n')
                    vtk.write('LOOKUP_TABLE default\n')
                    for dens in densities:
                        vtk.write(f'{dens:.6e}\n')
    
    # Print helpful summary
    print(f"✓ Exported {len(files)} grid files to {output_folder}/")
    print(f"  Open ParaView and load the .vtk files to visualize")
    
    if has_densities:
        print(f"  Tip: Color by 'density' to see density field evolution")
    if has_velocities:
        print(f"  Tip: Apply 'Glyph' filter with 'velocity' to visualize flow field")
        print(f"       Or use 'Stream Tracer' to show particle trajectories")

def export_surface_to_paraview(export_folder='vderm_exports',
                               subfolder='vderm_surface',
                               output_folder='paraview_surface'):
    """
    Export surface point clouds to VTK format for ParaView.
    
    Exports surface point clouds with all available attributes:
    normal vectors and density values (if present).
    
    Surface files are expected to have format: x y z n_x n_y n_z rho
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder
    subfolder : str, default='vderm_surface'
        Subfolder containing surface exports
    output_folder : str, default='paraview_surface'
        Folder to save VTK files
    
    Returns
    -------
    None
        VTK files saved to output_folder
    
    Examples
    --------
    >>> export_surface_to_paraview('my_exports')
    >>> # In ParaView: Load .vtk files
    >>> # Color by 'density' to see density distribution on surface
    >>> # Apply 'Glyph' filter to show normal vectors
    """
    
    pattern = os.path.join(export_folder, subfolder, '*.xyz')
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise FileNotFoundError(
            f"No surface files found at {pattern}\n"
            f"Make sure you ran run_VDERM_with_tracking with export_surface=True"
        )
    
    os.makedirs(output_folder, exist_ok=True)
    
    for f in tqdm(files, desc="Converting surface to VTK"):
        # Read surface data (should have positions + normals + densities)
        positions, normals, densities = read_xyz(f)
        
        # Check what data is available
        has_normals = normals is not None
        has_densities = densities is not None
        
        if not has_normals:
            warnings.warn(
                f"File {f} does not contain normal vectors. "
                "Surface exports should have 7 columns (x y z n_x n_y n_z rho)."
            )
        
        if not has_densities:
            warnings.warn(
                f"File {f} does not contain density data. "
                "Surface exports should have 7 columns (x y z n_x n_y n_z rho)."
            )
        
        # Create output filename
        basename = os.path.basename(f).replace('.xyz', '.vtk')
        output_path = os.path.join(output_folder, basename)
        
        # Write VTK file with all available attributes
        with open(output_path, 'w') as vtk:
            vtk.write('# vtk DataFile Version 3.0\n')
            vtk.write('VDERM Surface Point Cloud\n')
            vtk.write('ASCII\n')
            vtk.write('DATASET POLYDATA\n')
            
            # Write vertices
            vtk.write(f'POINTS {len(positions)} float\n')
            for pos in positions:
                vtk.write(f'{pos[0]:.6e} {pos[1]:.6e} {pos[2]:.6e}\n')
            
            # Write point data (only if we have normals or densities)
            if has_normals or has_densities:
                vtk.write(f'\nPOINT_DATA {len(positions)}\n')
                
                # Add normals as vector data
                if has_normals:
                    vtk.write('NORMALS normals float\n')
                    for norm in normals:
                        vtk.write(f'{norm[0]:.6e} {norm[1]:.6e} {norm[2]:.6e}\n')
                
                # Add density as scalar data
                if has_densities:
                    vtk.write('\nSCALARS density float 1\n')
                    vtk.write('LOOKUP_TABLE default\n')
                    for dens in densities:
                        vtk.write(f'{dens:.6e}\n')
    
    # Print summary
    print(f"✓ Exported {len(files)} surface files to {output_folder}/")
    print(f"  Open ParaView and load the .vtk files to visualize")
    
    if has_densities:
        print(f"  Tip: Color by 'density' to see density distribution on surface")
    if has_normals:
        print(f"  Tip: Apply 'Glyph' filter with 'normals' to visualize normal vectors")


def export_meshes_to_paraview(export_folder='vderm_exports',
                              subfolder='vderm_mesh',
                              output_folder='paraview_meshes'):
    """
    Convert STL meshes to VTK format for ParaView.
    
    **Note:** This function converts existing STL files to VTK but cannot include
    density data. For ParaView visualization with density vertex attributes,
    use run_VDERM_with_tracking() with mesh_format='vtk' instead.
    
    Exports reconstructed surface meshes with vertex normals. ParaView can render
    these with better quality than matplotlib and allows interactive exploration.
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder
    subfolder : str, default='vderm_mesh'
        Subfolder containing mesh exports (.stl files)
    output_folder : str, default='paraview_meshes'
        Folder to save VTK mesh files
    
    Returns
    -------
    None
        VTK files saved to output_folder
    
    See Also
    --------
    run_VDERM_with_tracking : Use mesh_format='vtk' for meshes with density data
    
    Examples
    --------
    >>> # Convert existing STL meshes to VTK (without density)
    >>> export_meshes_to_paraview('my_exports')
    >>> # In ParaView: File → Open → select .vtk files → Apply
    
    >>> # Preferred: Generate VTK meshes with density directly
    >>> run_VDERM_with_tracking(grid, surface_pts, normals,
    ...                         export_mesh=True,
    ...                         mesh_format='vtk')  # Better for ParaView!
    """
    
    # Print recommendation
    print("\n" + "="*70)
    print("NOTE: For ParaView visualization with density vertex attributes,")
    print("the recommended approach is to use run_VDERM_with_tracking()")
    print("with mesh_format='vtk' instead of converting STL files.")
    print()
    print("Example:")
    print("  run_VDERM_with_tracking(grid, surface_pts, normals,")
    print("                          export_mesh=True,")
    print("                          mesh_format='vtk')")
    print()
    print("This function converts existing STL → VTK but cannot include density.")
    print("="*70 + "\n")
        
    if not HAS_PYMESHLAB:
        raise ImportError(
            "This function requires pymeshlab for mesh conversion.\n"
            "Install with: pip install pymeshlab"
        )
    import pymeshlab as ml
    
    pattern = os.path.join(export_folder, subfolder, '*.stl')
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise FileNotFoundError(
            f"No mesh files found at {pattern}\n"
            f"Make sure you ran run_VDERM_with_tracking with export_mesh=True"
        )
    
    os.makedirs(output_folder, exist_ok=True)
    
    for f in tqdm(files, desc="Converting STL to VTK"):
        # Load mesh with pymeshlab
        ms = ml.MeshSet()
        ms.load_new_mesh(f)
        
        # Get mesh data
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        
        # Compute normals
        ms.compute_normal_for_point_clouds()
        normals = mesh.vertex_normal_matrix()
        
        # Create output filename
        basename = os.path.basename(f).replace('.stl', '.vtk')
        output_path = os.path.join(output_folder, basename)
        
        # Write VTK POLYDATA file (geometry + normals only, no density)
        with open(output_path, 'w') as vtk:
            vtk.write('# vtk DataFile Version 3.0\n')
            vtk.write('VDERM Mesh (converted from STL, no density data)\n')
            vtk.write('ASCII\n')
            vtk.write('DATASET POLYDATA\n')
            
            # Write vertices
            vtk.write(f'POINTS {len(vertices)} float\n')
            for v in vertices:
                vtk.write(f'{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n')
            
            # Write triangular faces
            vtk.write(f'\nPOLYGONS {len(faces)} {len(faces) * 4}\n')
            for face in faces:
                vtk.write(f'3 {face[0]} {face[1]} {face[2]}\n')
            
            # Write normals as vector data
            vtk.write(f'\nPOINT_DATA {len(vertices)}\n')
            vtk.write('NORMALS normals float\n')
            for n in normals:
                vtk.write(f'{n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
    
    print(f"\n✓ Exported {len(files)} mesh files to {output_folder}/")
    print(f"  Open ParaView and load the .vtk files to visualize")
    print(f"  Tip: In ParaView, use 'Surface With Edges' for best visualization")
    print(f"\n⚠ These meshes do NOT contain density data.")
    print(f"  To get density coloring in ParaView, use mesh_format='vtk'")
    print(f"  in run_VDERM_with_tracking() instead.\n")
    
def export_all_to_paraview(export_folder='vderm_exports',
                           output_base='paraview_exports'):
    """
    Export all available data (grid, surface, meshes) to ParaView format.
    
    Convenience function that exports everything that's available.
    
    **Note:** If meshes were exported as .stl files, they will be converted
    to VTK but will not contain density data. For meshes with density coloring
    in ParaView, use run_VDERM_with_tracking() with mesh_format='vtk'.
    
    Parameters
    ----------
    export_folder : str, default='vderm_exports'
        Base export folder from run_VDERM_with_tracking
    output_base : str, default='paraview_exports'
        Base folder for ParaView exports
    
    Examples
    --------
    >>> export_all_to_paraview('my_deformation')
    >>> # Creates paraview_exports/grid/, surface/, and meshes/
    """
    
    if not HAS_PYMESHLAB:
        raise ImportError(
            "This function requires pymeshlab for mesh conversion.\n"
            "Install with: pip install pymeshlab"
        )
    import pymeshlab as ml
    
    exported = []
    
    # Check what's available and export
    grid_path = os.path.join(export_folder, 'vderm_grid')
    if os.path.exists(grid_path) and len(glob.glob(os.path.join(grid_path, '*.xyz'))) > 0:
        output = os.path.join(output_base, 'grid')
        export_grid_to_paraview(export_folder, 'vderm_grid', output)
        exported.append('grid')
    
    surface_path = os.path.join(export_folder, 'vderm_surface')
    if os.path.exists(surface_path) and len(glob.glob(os.path.join(surface_path, '*.xyz'))) > 0:
        output = os.path.join(output_base, 'surface')
        export_surface_to_paraview(export_folder, 'vderm_surface', output)
        exported.append('surface')
    
    # Check for VTK meshes first, then STL
    mesh_path = os.path.join(export_folder, 'vderm_mesh')
    if os.path.exists(mesh_path):
        vtk_meshes = glob.glob(os.path.join(mesh_path, '*.vtk'))
        stl_meshes = glob.glob(os.path.join(mesh_path, '*.stl'))
        
        if len(vtk_meshes) > 0:
            # VTK meshes already exist - just copy them
            output = os.path.join(output_base, 'meshes')
            os.makedirs(output, exist_ok=True)
            import shutil
            for vtk_file in vtk_meshes:
                shutil.copy(vtk_file, output)
            print(f"✓ Copied {len(vtk_meshes)} VTK mesh files (with density data)")
            exported.append('meshes (VTK with density)')
            
        elif len(stl_meshes) > 0:
            # Convert STL to VTK (without density)
            output = os.path.join(output_base, 'meshes')
            export_meshes_to_paraview(export_folder, 'vderm_mesh', output)
            exported.append('meshes (converted from STL, no density)')
    
    if not exported:
        print("No VDERM exports found to convert!")
    else:
        print(f"\n✓ Exported {', '.join(exported)} to {output_base}/")
        print("\nTo visualize in ParaView:")
        print("  1. Open ParaView")
        print("  2. File → Open → Navigate to paraview_exports/")
        print("  3. Select all .vtk files in a folder")
        print("  4. Click 'Apply' in the Properties panel")
        print("  5. Use the time slider to animate through iterations")
        
        if 'meshes (converted from STL, no density)' in exported:
            print("\n⚠ Meshes converted from STL do not contain density data.")
            print("  For density coloring, re-run with mesh_format='vtk'")


def plot_pcd(positions, densities=None, normals=None, 
             title='Point Cloud',
             cmap='plasma',
             figsize=(12, 9),
             show_normals=False,
             normal_scale=0.05,
             point_size=1,
             alpha=0.6,
             view='3d',
             save_file=None):
    """
    Plot a single point cloud with optional density coloring and normal vectors.
    
    Can display as 3D view or three 2D orthogonal projections.
    
    Parameters
    ----------
    positions : ndarray, shape (n_points, 3)
        Point positions [x, y, z]
    densities : ndarray, shape (n_points,), optional
        Density values for color coding. If None, uses uniform color.
    normals : ndarray, shape (n_points, 3), optional
        Normal or velocity vectors to display
    title : str, default='Point Cloud'
        Plot title
    cmap : str, default='plasma'
        Colormap for density visualization
    figsize : tuple, default=(12, 9)
        Figure size in inches
    show_normals : bool, default=False
        If True and normals provided, draw vector arrows
    normal_scale : float, default=0.05
        Scale factor for normal vector arrows (as fraction of domain size)
    point_size : float, default=1
        Size of points in scatter plot
    alpha : float, default=0.6
        Point transparency (0=transparent, 1=opaque)
    view : str, default='3d'
        View type: '3d' for 3D plot, '2d' for three orthogonal projections
    save_file : str, optional
        If provided, save plot to this file instead of showing
    
    Returns
    -------
    fig : matplotlib Figure
        The figure object
    
    Examples
    --------
    >>> # Simple 3D plot
    >>> positions = np.random.rand(1000, 3)
    >>> plot_pcd(positions)
    
    >>> # With density coloring
    >>> densities = np.random.rand(1000)
    >>> plot_pcd(positions, densities=densities, cmap='viridis')
    
    >>> # 2D projections with density
    >>> plot_pcd(positions, densities=densities, view='2d')
    
    >>> # Show normal vectors
    >>> normals = np.random.randn(1000, 3)
    >>> plot_pcd(positions, normals=normals, show_normals=True)
    
    >>> # Save to file
    >>> plot_pcd(positions, densities=densities, save_file='myplot.png')
    """
    
    # Get bounds
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    ranges = pos_max - pos_min
    
    # Compute normal scale if showing normals
    if show_normals and normals is not None:
        arrow_length = normal_scale * ranges.mean()
    
    if view == '3d':
        # Single 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        if densities is not None:
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=densities, cmap=cmap, s=point_size, alpha=alpha)
            cbar = plt.colorbar(scatter, ax=ax, label='Density', shrink=0.6)
        else:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='dodgerblue', s=point_size, alpha=alpha)
        
        # Show normals if requested
        if show_normals and normals is not None:
            # Subsample for clarity
            step = max(1, len(positions) // 100)
            ax.quiver(positions[::step, 0], positions[::step, 1], positions[::step, 2],
                     normals[::step, 0], normals[::step, 1], normals[::step, 2],
                     length=arrow_length, color='red', alpha=0.5, arrow_length_ratio=0.3)
        
        # Set limits and aspect
        ax.set_xlim(pos_min[0], pos_max[0])
        ax.set_ylim(pos_min[1], pos_max[1])
        ax.set_zlim(pos_min[2], pos_max[2])
        ax.set_box_aspect(ranges)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
    elif view == '2d':
        # Three orthogonal projections
        fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=figsize)
        
        # XY plane
        if densities is not None:
            scatter = ax_xy.scatter(positions[:, 0], positions[:, 1],
                                   c=densities, cmap=cmap, s=point_size, alpha=alpha)
        else:
            ax_xy.scatter(positions[:, 0], positions[:, 1],
                         c='dodgerblue', s=point_size, alpha=alpha)
        
        ax_xy.set_xlim(pos_min[0], pos_max[0])
        ax_xy.set_ylim(pos_min[1], pos_max[1])
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.set_title('XY Plane (Top View)')
        ax_xy.set_aspect('equal')
        ax_xy.grid(True, alpha=0.3)
        
        # XZ plane
        if densities is not None:
            ax_xz.scatter(positions[:, 0], positions[:, 2],
                         c=densities, cmap=cmap, s=point_size, alpha=alpha)
        else:
            ax_xz.scatter(positions[:, 0], positions[:, 2],
                         c='dodgerblue', s=point_size, alpha=alpha)
        
        ax_xz.set_xlim(pos_min[0], pos_max[0])
        ax_xz.set_ylim(pos_min[2], pos_max[2])
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title('XZ Plane (Front View)')
        ax_xz.set_aspect('equal')
        ax_xz.grid(True, alpha=0.3)
        
        # YZ plane
        if densities is not None:
            scatter = ax_yz.scatter(positions[:, 1], positions[:, 2],
                                   c=densities, cmap=cmap, s=point_size, alpha=alpha)
        else:
            ax_yz.scatter(positions[:, 1], positions[:, 2],
                         c='dodgerblue', s=point_size, alpha=alpha)
        
        ax_yz.set_xlim(pos_min[1], pos_max[1])
        ax_yz.set_ylim(pos_min[2], pos_max[2])
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.set_title('YZ Plane (Side View)')
        ax_yz.set_aspect('equal')
        ax_yz.grid(True, alpha=0.3)
        
        # Add colorbar if using densities
        if densities is not None:
            cbar = fig.colorbar(scatter, ax=[ax_xy, ax_xz, ax_yz],
                              label='Density', shrink=0.8, pad=0.02)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        # plt.tight_layout()
        
    else:
        raise ValueError(f"view must be '3d' or '2d', got '{view}'")
    
    # Save or show
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved to: {save_file}")
    else:
        plt.show()
    
    return fig