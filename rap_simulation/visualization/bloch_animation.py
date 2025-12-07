"""
Bloch sphere visualization and animation using QuTiP.

This module uses QuTiP's built-in Bloch class for sphere visualization
and provides animation capabilities for showing state evolution.
"""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from qutip import Bloch

if TYPE_CHECKING:
    from ..simulation import SimulationResult


def plot_bloch_trajectory(
    result: "SimulationResult",
    show_states: bool = True,
    show_trajectory: bool = True,
    show_initial_final: bool = True,
    figsize: tuple[float, float] = (8, 8),
    show: bool = True,
) -> tuple[Figure, Bloch]:
    """
    Plot the state trajectory on the Bloch sphere.
    
    Uses QuTiP's Bloch class for visualization.
    
    Args:
        result: Simulation result object.
        show_states: Whether to show individual state points.
        show_trajectory: Whether to show the trajectory line.
        show_initial_final: Whether to highlight initial and final states.
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Tuple of (Figure, Bloch) objects.
    """
    fig = plt.figure(figsize=figsize)
    
    b = Bloch(fig=fig)
    
    # Customize appearance
    b.vector_color = ['#e74c3c']
    b.point_color = ['#3498db']
    b.point_marker = ['o']
    b.point_size = [15]
    
    # Get coordinates
    x = result.bloch_coords['x']
    y = result.bloch_coords['y']
    z = result.bloch_coords['z']
    
    if show_trajectory:
        # Add trajectory as points (QuTiP Bloch uses points for trajectories)
        # Subsample for clarity if too many points
        n_points = len(x)
        step = max(1, n_points // 200)
        
        b.add_points([x[::step], y[::step], z[::step]], meth='l')
    
    if show_states:
        # Add some state points along the trajectory
        n_show = 20
        indices = np.linspace(0, len(x)-1, n_show, dtype=int)
        b.add_points([x[indices], y[indices], z[indices]])
    
    if show_initial_final:
        # Mark initial state (should be near north pole)
        b.add_vectors([[x[0], y[0], z[0]]])
        # Mark final state
        b.add_vectors([[x[-1], y[-1], z[-1]]])
    
    b.make_sphere()
    
    # Add custom labels
    ax = fig.axes[0]
    ax.set_title(f'Bloch Sphere Trajectory\nFinal P₁: {result.final_p1:.3f}', fontsize=12)
    
    if show:
        plt.show()
    
    return fig, b


def plot_bloch_with_driving_field(
    result: "SimulationResult",
    driving_coords: dict | None = None,
    figsize: tuple[float, float] = (8, 8),
    show: bool = True,
) -> tuple[Figure, Bloch]:
    """
    Plot state trajectory with the effective driving field direction.
    
    Args:
        result: Simulation result object.
        driving_coords: Driving field coordinates from compute_driving_field_trajectory().
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Tuple of (Figure, Bloch) objects.
    """
    from ..analysis import compute_driving_field_trajectory
    
    fig = plt.figure(figsize=figsize)
    b = Bloch(fig=fig)
    
    # State trajectory
    x = result.bloch_coords['x']
    y = result.bloch_coords['y']
    z = result.bloch_coords['z']
    
    step = max(1, len(x) // 200)
    b.add_points([x[::step], y[::step], z[::step]], meth='l')
    
    # Driving field trajectory (if provided)
    if driving_coords is not None:
        xd = driving_coords['x']
        yd = driving_coords['y']
        zd = driving_coords['z']
        
        step_d = max(1, len(xd) // 100)
        # Use different color for driving field
        b.point_color = ['#3498db', '#e74c3c']
        b.add_points([xd[::step_d], yd[::step_d], zd[::step_d]], meth='l')
    
    b.make_sphere()
    
    if show:
        plt.show()
    
    return fig, b


def animate_bloch(
    result: "SimulationResult",
    fps: int = 30,
    duration: float = 5.0,
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    show: bool = True,
) -> FuncAnimation:
    """
    Create an animation of the state evolution on the Bloch sphere.
    
    Uses QuTiP's Bloch class with matplotlib animation.
    
    Args:
        result: Simulation result object.
        fps: Frames per second for animation.
        duration: Total animation duration in seconds.
        save_path: If provided, save animation to this file path (.gif or .mp4).
        figsize: Figure size.
        show: Whether to call plt.show() after creating animation.
        
    Returns:
        Matplotlib FuncAnimation object.
    """
    n_frames = int(fps * duration)
    n_points = len(result.times)
    frame_indices = np.linspace(0, n_points - 1, n_frames, dtype=int)
    
    # Get coordinates
    x = result.bloch_coords['x']
    y = result.bloch_coords['y']
    z = result.bloch_coords['z']
    
    # Create figure and Bloch sphere
    fig = plt.figure(figsize=figsize)
    
    def make_frame(frame_idx):
        """Create a single frame of the animation."""
        fig.clear()
        
        b = Bloch(fig=fig)
        
        idx = frame_indices[frame_idx]
        
        # Add trajectory up to current point
        traj_idx = np.linspace(0, idx, min(idx + 1, 100), dtype=int)
        if len(traj_idx) > 1:
            b.add_points([x[traj_idx], y[traj_idx], z[traj_idx]], meth='l')
        
        # Current state vector
        b.add_vectors([[x[idx], y[idx], z[idx]]])
        
        b.make_sphere()
        
        # Add time label
        ax = fig.axes[0]
        current_time = result.times[idx] * 1e3  # Convert to ms
        current_p1 = result.probabilities['p1'][idx]
        ax.set_title(f't = {current_time:.3f} ms\nP₁ = {current_p1:.3f}', fontsize=12)
        
        return fig,
    
    ani = FuncAnimation(
        fig, 
        make_frame, 
        frames=n_frames,
        interval=1000/fps,
        blit=False,
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        else:
            ani.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    if show:
        plt.show()
    
    return ani


def animate_bloch_with_precession(
    result: "SimulationResult",
    fps: int = 30,
    duration: float = 5.0,
    show_driving_field: bool = True,
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    show: bool = True,
) -> FuncAnimation:
    """
    Animate the Bloch sphere showing both state vector and effective field.
    
    This visualization helps understand how the state follows (or doesn't follow)
    the adiabatically changing driving field direction.
    
    Args:
        result: Simulation result object.
        fps: Frames per second.
        duration: Animation duration in seconds.
        show_driving_field: Whether to show the effective driving field vector.
        save_path: Optional path to save the animation.
        figsize: Figure size.
        show: Whether to show the animation.
        
    Returns:
        FuncAnimation object.
    """
    from ..analysis import compute_driving_field_trajectory
    
    n_frames = int(fps * duration)
    n_points = len(result.times)
    frame_indices = np.linspace(0, n_points - 1, n_frames, dtype=int)
    
    # State coordinates
    x = result.bloch_coords['x']
    y = result.bloch_coords['y']
    z = result.bloch_coords['z']
    
    # Driving field coordinates
    if show_driving_field:
        driving = compute_driving_field_trajectory(
            result.params,
            result.pulse_name,
            result.detuning_name,
        )
        xd, yd, zd = driving['x'], driving['y'], driving['z']
    
    fig = plt.figure(figsize=figsize)
    
    def make_frame(frame_idx):
        fig.clear()
        
        b = Bloch(fig=fig)
        b.vector_color = ['#e74c3c', '#3498db']  # State: red, Field: blue
        
        idx = frame_indices[frame_idx]
        
        # Trajectory up to current point
        traj_idx = np.linspace(0, idx, min(idx + 1, 100), dtype=int)
        if len(traj_idx) > 1:
            b.add_points([x[traj_idx], y[traj_idx], z[traj_idx]], meth='l')
        
        # Current state vector
        b.add_vectors([[x[idx], y[idx], z[idx]]])
        
        # Driving field vector
        if show_driving_field:
            b.add_vectors([[xd[idx], yd[idx], zd[idx]]])
        
        b.make_sphere()
        
        # Labels
        ax = fig.axes[0]
        current_time = result.times[idx] * 1e3
        current_p1 = result.probabilities['p1'][idx]
        ax.set_title(
            f't = {current_time:.3f} ms | P₁ = {current_p1:.3f}\n'
            f'Red: State | Blue: Driving Field',
            fontsize=11
        )
        
        return fig,
    
    ani = FuncAnimation(
        fig,
        make_frame,
        frames=n_frames,
        interval=1000/fps,
        blit=False,
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        else:
            ani.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    if show:
        plt.show()
    
    return ani


def create_bloch_snapshot(
    result: "SimulationResult",
    time_index: int | None = None,
    time_value: float | None = None,
    figsize: tuple[float, float] = (6, 6),
    show: bool = True,
) -> tuple[Figure, Bloch]:
    """
    Create a snapshot of the Bloch sphere at a specific time.
    
    Args:
        result: Simulation result.
        time_index: Index into the time array.
        time_value: Time value in seconds (will find nearest index).
        figsize: Figure size.
        show: Whether to show.
        
    Returns:
        Tuple of (Figure, Bloch).
    """
    if time_index is None and time_value is not None:
        time_index = np.argmin(np.abs(result.times - time_value))
    elif time_index is None:
        time_index = -1  # Default to final state
    
    fig = plt.figure(figsize=figsize)
    b = Bloch(fig=fig)
    
    x = result.bloch_coords['x'][time_index]
    y = result.bloch_coords['y'][time_index]
    z = result.bloch_coords['z'][time_index]
    
    b.add_vectors([[x, y, z]])
    b.make_sphere()
    
    ax = fig.axes[0]
    t_ms = result.times[time_index] * 1e3
    p1 = result.probabilities['p1'][time_index]
    ax.set_title(f't = {t_ms:.3f} ms | P₁ = {p1:.3f}', fontsize=12)
    
    if show:
        plt.show()
    
    return fig, b

