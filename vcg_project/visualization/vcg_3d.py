"""
3D VCG visualization using Plotly.

Creates animated 3D plots of vectorcardiogram loops with
comparison between methods (Kors, LSTM, Transformer, Ground Truth).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


# Color scheme for each method
METHOD_COLORS = {
    "Ground Truth": "#2ecc71",  # green
    "Kors": "#3498db",          # blue
    "LSTM": "#e67e22",          # orange
    "Transformer": "#e74c3c",   # red
}


def create_vcg_3d_static(
    gt_xyz: np.ndarray,
    kors_xyz: np.ndarray | None = None,
    lstm_xyz: np.ndarray | None = None,
    transformer_xyz: np.ndarray | None = None,
    title: str = "VCG Loop Comparison",
) -> go.Figure:
    """
    Create a static 3D VCG plot comparing all methods.

    Parameters
    ----------
    gt_xyz : np.ndarray
        Ground truth VCG, shape [T, 3] or [3, T].
    kors_xyz : np.ndarray, optional
        Kors transform output.
    lstm_xyz : np.ndarray, optional
        LSTM model output.
    transformer_xyz : np.ndarray, optional
        Transformer model output.
    title : str
        Plot title.

    Returns
    -------
    go.Figure
        Plotly 3D scatter/line figure.
    """
    fig = go.Figure()

    traces = [
        ("Ground Truth", gt_xyz, 4),
        ("Kors", kors_xyz, 2),
        ("LSTM", lstm_xyz, 2),
        ("Transformer", transformer_xyz, 2),
    ]

    for name, data, width in traces:
        if data is None:
            continue
        xyz = _to_T3(data)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines",
            line=dict(
                color=METHOD_COLORS[name],
                width=width,
            ),
            name=name,
            opacity=0.9 if name == "Ground Truth" else 0.7,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (left-right)",
            yaxis_title="Y (superior-inferior)",
            zaxis_title="Z (anterior-posterior)",
            aspectmode="data",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        width=800,
        height=700,
    )

    return fig


def create_vcg_3d_animated(
    gt_xyz: np.ndarray,
    kors_xyz: np.ndarray | None = None,
    lstm_xyz: np.ndarray | None = None,
    transformer_xyz: np.ndarray | None = None,
    title: str = "VCG Loop Animation",
    frame_step: int | None = None,
    target_frames: int = 60,
    frame_duration: int = 30,
) -> go.Figure:
    """
    Create an animated 3D VCG plot that builds up over time.

    Parameters
    ----------
    gt_xyz : np.ndarray
        Ground truth VCG, shape [T, 3] or [3, T].
    kors_xyz, lstm_xyz, transformer_xyz : np.ndarray, optional
        Method outputs.
    title : str
        Plot title.
    frame_step : int, optional
        Number of time points per animation frame. If None (default),
        it's derived from the beat length so the animation always has
        about `target_frames` frames regardless of the underlying
        sample rate — e.g. raw ~1000-sample beats and resampled
        256-sample beats both render at a similar, smooth frame count
        instead of the raw beat producing ~4x more (choppier, slower
        to render) frames.
    target_frames : int
        Desired number of animation frames when `frame_step` is None.
    frame_duration : int
        Milliseconds between frames.

    Returns
    -------
    go.Figure
        Animated Plotly figure with play/pause controls.
    """
    gt = _to_T3(gt_xyz)
    T = len(gt)

    if frame_step is None:
        frame_step = max(1, T // target_frames)

    # Build list of available methods
    methods = [("Ground Truth", gt, METHOD_COLORS["Ground Truth"])]
    if kors_xyz is not None:
        methods.append(("Kors", _to_T3(kors_xyz), METHOD_COLORS["Kors"]))
    if lstm_xyz is not None:
        methods.append(("LSTM", _to_T3(lstm_xyz), METHOD_COLORS["LSTM"]))
    if transformer_xyz is not None:
        methods.append(("Transformer", _to_T3(transformer_xyz), METHOD_COLORS["Transformer"]))

    # Create frames
    n_frames = T // frame_step
    frames = []

    for i in range(n_frames):
        end_idx = (i + 1) * frame_step
        frame_data = []
        for name, xyz, color in methods:
            frame_data.append(go.Scatter3d(
                x=xyz[:end_idx, 0],
                y=xyz[:end_idx, 1],
                z=xyz[:end_idx, 2],
                mode="lines",
                line=dict(color=color, width=3),
                name=name,
            ))
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Initial frame (empty or first few points)
    initial_data = []
    for name, xyz, color in methods:
        initial_data.append(go.Scatter3d(
            x=xyz[:frame_step, 0],
            y=xyz[:frame_step, 1],
            z=xyz[:frame_step, 2],
            mode="lines",
            line=dict(color=color, width=3),
            name=name,
        ))

    fig = go.Figure(
        data=initial_data,
        frames=frames,
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (left-right)",
            yaxis_title="Y (superior-inferior)",
            zaxis_title="Z (anterior-posterior)",
            aspectmode="data",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.05,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Time: "),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                xanchor="left",
                y=0,
                yanchor="top",
                steps=[
                    dict(
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                        label=str(i),
                        method="animate",
                    )
                    for i in range(n_frames)
                ],
            )
        ],
        width=800,
        height=700,
    )

    return fig


def create_vcg_2d_projections(
    gt_xyz: np.ndarray,
    kors_xyz: np.ndarray | None = None,
    lstm_xyz: np.ndarray | None = None,
    transformer_xyz: np.ndarray | None = None,
    title: str = "VCG 2D Projections",
) -> go.Figure:
    """
    Create 2D projection plots (Frontal, Sagittal, Transverse planes).

    Parameters
    ----------
    gt_xyz : np.ndarray
        Ground truth VCG, shape [T, 3] or [3, T].

    Returns
    -------
    go.Figure
        Plotly figure with 3 subplots.
    """
    from plotly.subplots import make_subplots

    gt = _to_T3(gt_xyz)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Frontal (X-Y)", "Sagittal (Y-Z)", "Transverse (X-Z)"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
    )

    projections = [
        (0, 1, "X", "Y", 1, 1),  # Frontal
        (1, 2, "Y", "Z", 1, 2),  # Sagittal
        (0, 2, "X", "Z", 1, 3),  # Transverse
    ]

    methods = [
        ("Ground Truth", gt, METHOD_COLORS["Ground Truth"], 2.5),
        ("Kors", _to_T3(kors_xyz) if kors_xyz is not None else None, METHOD_COLORS["Kors"], 1.5),
        ("LSTM", _to_T3(lstm_xyz) if lstm_xyz is not None else None, METHOD_COLORS["LSTM"], 1.5),
        ("Transformer", _to_T3(transformer_xyz) if transformer_xyz is not None else None, METHOD_COLORS["Transformer"], 1.5),
    ]

    for xi, yi, xlab, ylab, row, col in projections:
        for name, xyz, color, width in methods:
            if xyz is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xyz[:, xi],
                    y=xyz[:, yi],
                    mode="lines",
                    line=dict(color=color, width=width),
                    name=name,
                    showlegend=(row == 1 and col == 1),
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=title,
        width=1200,
        height=400,
    )

    # Set axis labels
    for i, (xi, yi, xlab, ylab, _, _) in enumerate(projections):
        fig.update_xaxes(title_text=xlab, row=1, col=i+1)
        fig.update_yaxes(title_text=ylab, row=1, col=i+1, scaleanchor=f"x{i+1}" if i > 0 else None)

    return fig


def _to_T3(data: np.ndarray) -> np.ndarray:
    """
    Ensure data is in [T, 3] format.

    Handles both [T, 3] and [3, T] inputs.
    """
    if data.ndim == 2:
        if data.shape[0] == 3 and data.shape[1] != 3:
            return data.T
        return data
    raise ValueError(f"Expected 2D array, got shape {data.shape}")
