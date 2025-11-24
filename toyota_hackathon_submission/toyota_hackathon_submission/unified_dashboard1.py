from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import track_projection

# ------------------------
# Configuration & Track Definitions
# ------------------------
st.set_page_config(layout="wide", page_title="Toyota Racing Analysis")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TOYOTA_GR_DIR = BASE_DIR.parent / "TOYOTA GR"

# Track Configs
TRACK_CONFIG = {
    'COTA': {
        'type': 'pixel',
        'endurance_file': str(TOYOTA_GR_DIR / "COTA" / "Race 1" / "23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV"),
        'telemetry_file': str(TOYOTA_GR_DIR / "COTA" / "Race 1" / "R1_cota_telemetry_data.csv"),
        'micro_sectors': [
            ('S1-a (T1)', 'IM1a_time'),
            ('S1-b (T2-T7)', 'IM1_time'),
            ('S2-a (T8-T11)', 'IM2a_time'),
            ('S2-b (T12-T15)', 'IM2_time'),
            ('S3-a (T16-T19)', 'IM3a_time'),
            ('S3-b (T20-S/F)', 'FL_time')
        ],
        'sector_annotations': {
            'S1': ((350, 600), 'S1'),
            'S2': ((800, 350), 'S2'),
            'S3': ((250, 400), 'S3'),
        },
        'subsection_annotations': {
            'S1-a': ((310, 670), 'S1-a'),
            'S1-b': ((500, 480), 'S1-b'),
            'S2-a': ((700, 365), 'S2-a'), 
            'S2-b': ((450, 300), 'S2-b'), 
            'S3-a': ((200, 480), 'S3-a'),
            'S3-b': ((80, 450), 'S3-b'),
        }
    },
    'Barber': {
        'type': 'gps',
        'endurance_file': str(TOYOTA_GR_DIR / "barber" / "23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV"),
        'telemetry_file': str(TOYOTA_GR_DIR / "barber" / "R1_barber_telemetry_data.csv"),
        'telemetry_ref_file': str(DATA_DIR / "barber_reference_lap.csv"),
        'micro_sectors': [
            ('S1-a', 'IM1a_time'),
            ('S1-b', 'IM1_time'),
            ('S2-a', 'IM2a_time'),
            ('S2-b', 'IM2_time'),
            ('S3-a', 'IM3a_time'),
            ('S3-b', 'FL_time')
        ],
        'sector_annotations': {}, 
        'subsection_annotations': {}
    }
}

# Barber Turns (Distance-based)
# Barber Turns (Distance-based) - DEPRECATED, will use Auto-Detection
# BARBER_TURNS = { ... }

def detect_turns(ref_df, min_speed_dip=10, window_size=100):
    """
    Automatically detects turns based on speed minima in the reference lap.
    Returns a dictionary of {TurnName: (start_dist, end_dist)}.
    """
    if ref_df.empty: return {}
    
    # Smooth speed
    speed = savgol_filter(ref_df['speed'], 51, 3)
    dist = ref_df['dist'].values
    
    # Find local minima
    from scipy.signal import argrelextrema
    # Find indices where speed is a local minimum
    # We use a window to avoid noise
    # Relaxed order to 25 (approx 25m) to catch tight sequences like T6-T7
    local_min_idxs = argrelextrema(speed, np.less, order=25)[0] 
    
    turns = {}
    turn_count = 1
    
    for idx in local_min_idxs:
        apex_speed = speed[idx]
        apex_dist = dist[idx]
        
        # Filter: Must be a significant corner (e.g. speed < 200 km/h or significant drop)
        # For now, just take all distinct minima as "Turns"
        
        # Define Window
        start_dist = max(0, apex_dist - window_size)
        end_dist = min(dist[-1], apex_dist + window_size)
        
        turns[f"Turn {turn_count}"] = (start_dist, end_dist)
        turn_count += 1
        
    return turns

def merge_turns(turns_dict, merge_threshold=400):
    """Merges turns that are close to each other."""
    if not turns_dict: return {}
    
    sorted_turns = sorted(turns_dict.items(), key=lambda x: x[1][0])
    merged = {}
    
    current_name, (current_start, current_end) = sorted_turns[0]
    
    for i in range(1, len(sorted_turns)):
        next_name, (next_start, next_end) = sorted_turns[i]
        
        if next_start - current_end < merge_threshold:
            # Merge
            current_end = max(current_end, next_end)
            current_name = f"{current_name}-{next_name.split(' ')[-1]}" # e.g. Turn 1-2
        else:
            # Save current and start new
            merged[current_name] = (current_start, current_end)
            current_name = next_name
            current_start = next_start
            current_end = next_end
            
    merged[current_name] = (current_start, current_end)
    return merged

# ------------------------
# Simulation helpers (fastest-lap playback)
# ------------------------

SIM_CONFIG = {
    "COTA": {
        "fastest_laps": Path(BASE_DIR.parent / "fastest_laps.parquet"),
        "fastest_telemetry": Path(BASE_DIR.parent / "fastest_lap_telemetry.parquet"),
        "layout": None,
    },
    "Barber": {
        "fastest_laps": Path(BASE_DIR.parent / "barber_fastest_laps.parquet"),
        "fastest_telemetry": Path(BASE_DIR.parent / "barber_fastest_lap_telemetry.parquet"),
        "layout": Path(DATA_DIR / "barber_reference_lap.csv")
    },
}

def cota_polyline() -> np.ndarray:
    """Approximate COTA centerline for simulation playback."""
    ref_points = [
        (98, 466, 115, 460),
        (290, 660, 303, 644),
        (303, 644, 318, 662),
        (303, 644, 322, 641),
        (290, 576, 307, 576),
        (301, 549, 314, 560),
        (319, 527, 326, 542),
        (384, 499, 395, 512),
        (396, 494, 410, 508),
        (404, 481, 421, 491),
        (409, 467, 427, 477),
        (424, 459, 436, 469),
        (441, 449, 449, 464),
        (441, 449, 449, 464),
        (457, 442, 470, 454),
        (463, 430, 481, 437),
        (470, 406, 484, 418),
        (517, 337, 521, 397),
        (544, 384, 532, 401),
        (566, 400, 558, 412),
        (582, 410, 582, 428),
        (596, 401, 608, 415),
        (614, 390, 628, 401),
        (643, 377, 642, 392),
        (667, 392, 652, 402),
        (667, 392, 652, 402),
        (670, 400, 660, 413),
        (681, 403, 685, 420),
        (740, 398, 740, 416),
        (752, 400, 760, 415),
        (762, 389, 777, 400),
        (809, 310, 830, 318),
        (815, 303, 836, 294),
        (807, 306, 805, 287),
        (341, 334, 341, 317),
        (327, 335, 308, 322),
        (333, 347, 315, 354),
        (357, 399, 337, 399),
        (359, 421, 342, 411),
        (332, 428, 334, 411),
        (319, 423, 322, 407),
        (303, 416, 316, 404),
        (297, 399, 314, 396),
        (285, 375, 303, 370),
        (281, 367, 293, 355),
        (277, 368, 275, 350),
        (277, 368, 275, 350),
        (269, 366, 253, 351),
        (274, 379, 256, 382),
        (292, 427, 280, 446),
        (280, 490, 266, 475),
        (256, 497, 250, 481),
        (256, 497, 250, 481),
        (197, 495, 206, 479),
        (166, 461, 182, 449),
        (143, 417, 157, 406),
        (131, 400, 136, 384),
        (115, 403, 106, 388),
        (95, 405, 80, 390),
        (80, 420, 60, 400),
        (75, 440, 95, 425),
        (98, 466, 115, 460),
    ]
    centers = np.array([((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in ref_points])
    min_xy = centers.min(axis=0)
    max_xy = centers.max(axis=0)
    norm = (centers - min_xy) / (max_xy - min_xy)
    norm[:, 1] = 1.0 - norm[:, 1]
    return norm

def barber_polyline(layout_path: Path) -> np.ndarray:
    if layout_path and layout_path.exists():
        df = pd.read_csv(layout_path)
        # Support multiple schema variants for track layout
        if {"x_m", "y_m"}.issubset(df.columns):
            pts = df[["x_m", "y_m"]].to_numpy()
        elif {"x", "y"}.issubset(df.columns):
            pts = df[["x", "y"]].to_numpy()
        else:
            # Fallback: pick first two numeric columns
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if len(numeric_cols) >= 2:
                pts = df[numeric_cols[:2]].to_numpy()
            else:
                pts = None
        if pts is not None and len(pts) > 1:
            # Drop non-finite rows to keep the polyline valid
            mask = np.isfinite(pts).all(axis=1)
            pts = pts[mask]
        if pts is not None and len(pts) > 1:
            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            norm = (pts - min_xy) / (max_xy - min_xy)
            return norm
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    x = 0.5 + 0.45 * np.cos(angles)
    y = 0.5 + 0.25 * np.sin(angles)
    return np.stack([x, y], axis=1)

@st.cache_data
def load_fastest_assets(track_name: str):
    cfg = SIM_CONFIG.get(track_name)
    if not cfg:
        return pd.DataFrame(), pd.DataFrame(), None
    laps_path = cfg["fastest_laps"]
    tele_path = cfg["fastest_telemetry"]
    layout = cfg["layout"]

    def _read(path: Path):
        if path and path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    return pd.read_csv(csv_path)
        return pd.DataFrame()

    fastest = _read(laps_path)
    tele = _read(tele_path)
    if track_name == "COTA":
        poly = cota_polyline()
    else:
        # Prefer the Barber reference lap-derived track for consistency with the main dashboard
        ref_path = Path(TRACK_CONFIG["Barber"].get("telemetry_ref_file", ""))
        centerline_points = []
        if ref_path.exists():
            centerline_points, _ = load_barber_track(str(ref_path))
        if centerline_points:
            pts = np.array(centerline_points)
            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            pts = (pts - min_xy) / (max_xy - min_xy)
            poly = pts
        else:
            poly = barber_polyline(layout)
    return fastest, tele, poly

def build_sim_frames(fastest_df: pd.DataFrame, tele_df: pd.DataFrame, car_numbers, polyline: np.ndarray):
    """Slice telemetry for selected fastest laps and compute XY frames driven by raw telemetry."""
    if tele_df.empty or fastest_df.empty or polyline is None:
        return pd.DataFrame()
    frames = []
    diffs = np.diff(polyline, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    cum /= cum[-1]

    def interp_track(progress):
        progress = np.clip(progress, 0, 1)
        x = np.interp(progress, cum, polyline[:, 0])
        y = np.interp(progress, cum, polyline[:, 1])
        return x, y

    tele_df = tele_df.copy()
    if "vehicle_number" in tele_df.columns:
        tele_df["vehicle_number"] = pd.to_numeric(tele_df["vehicle_number"], errors="coerce")
    if "vehicle_id" in tele_df.columns:
        tele_df["vehicle_id"] = tele_df["vehicle_id"].astype(str)

    for car in car_numbers:
        row = fastest_df[fastest_df["CAR_NUMBER"] == car]
        if row.empty:
            continue
        row = row.iloc[0]
        tele_car = tele_df.loc[tele_df["vehicle_number"] == car].copy() if "vehicle_number" in tele_df.columns else pd.DataFrame()
        if tele_car.empty and "VEHICLE_ID" in row and pd.notna(row["VEHICLE_ID"]) and "vehicle_id" in tele_df.columns:
            tele_car = tele_df.loc[tele_df["vehicle_id"].astype(str) == str(row["VEHICLE_ID"])].copy()
        if tele_car.empty:
            continue

        if "ts" in tele_car.columns:
            tele_car.loc[:, "ts"] = pd.to_datetime(tele_car["ts"], errors="coerce")
        else:
            ts = pd.to_datetime(tele_car.get("timestamp"), errors="coerce")
            if "meta_time" in tele_car.columns:
                ts = ts.fillna(pd.to_datetime(tele_car["meta_time"], errors="coerce"))
            tele_car.loc[:, "ts"] = ts
        tele_car = tele_car.dropna(subset=["ts"]).sort_values("ts").copy()
        if tele_car.empty:
            continue
        tele_car["elapsed_s"] = (tele_car["ts"] - tele_car["ts"].iloc[0]).dt.total_seconds()
        tele_car = tele_car.sort_values("elapsed_s")
        speed_mask = tele_car["telemetry_name"].str.lower() == "speed"
        tele_car["speed_val"] = np.nan
        tele_car.loc[speed_mask, "speed_val"] = tele_car.loc[speed_mask, "telemetry_value"]
        tele_car["speed_val"] = tele_car["speed_val"].ffill().bfill().fillna(0)
        tele_car["dt"] = tele_car["elapsed_s"].diff().fillna(0)
        tele_car["speed_mps"] = tele_car["speed_val"] * (1000 / 3600)
        tele_car["dist_m"] = (tele_car["speed_mps"] * tele_car["dt"]).cumsum()
        dist_max = tele_car["dist_m"].max()
        if pd.notna(dist_max) and dist_max > 0:
            tele_car["progress"] = np.clip(tele_car["dist_m"] / dist_max, 0, 1)
        else:
            span = tele_car["elapsed_s"].max()
            tele_car["progress"] = np.clip(tele_car["elapsed_s"] / span if span else 0, 0, 1)
        x, y = interp_track(tele_car["progress"].to_numpy())
        tele_car["x"] = x
        tele_car["y"] = y
        tele_car["frame"] = tele_car["elapsed_s"].round(1)
        frames.append(tele_car[["vehicle_number", "frame", "x", "y", "speed_val", "elapsed_s"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def simulation_plot(sim_df: pd.DataFrame, polyline: np.ndarray, sector_fracs: tuple):
    """Plotly animation of fastest-lap telemetry with sector-colored track."""
    if sim_df.empty or polyline is None:
        return go.Figure()
    track = polyline.copy()
    s1_frac, s2_frac = sector_fracs
    if not np.isfinite(s1_frac) or not np.isfinite(s2_frac):
        s1_frac, s2_frac = 0.33, 0.66
    s1_frac = min(max(s1_frac, 1e-3), 0.995)
    s2_frac = min(max(s2_frac, s1_frac + 1e-3), 0.999)
    if (s1_frac < 0.02) or (s2_frac - s1_frac < 0.02) or (1 - s2_frac < 0.02):
        s1_frac, s2_frac = 0.33, 0.66

    tmin = track.min(axis=0)
    tmax = track.max(axis=0)
    track = (track - tmin) / (tmax - tmin)
    sim_df = sim_df.copy()
    sim_df["x"] = (sim_df["x"] - tmin[0]) / (tmax[0] - tmin[0])
    sim_df["y"] = (sim_df["y"] - tmin[1]) / (tmax[1] - tmin[1])
    sim_df["x"] = sim_df["x"].clip(0, 1)
    sim_df["y"] = sim_df["y"].clip(0, 1)

    progress = np.linspace(0, 1, 400)
    tx = np.interp(progress, np.linspace(0, 1, len(track)), track[:, 0])
    ty = np.interp(progress, np.linspace(0, 1, len(track)), track[:, 1])
    n = len(progress)
    idx1 = max(1, int(round(s1_frac * (n - 1))))
    idx2 = max(idx1 + 1, int(round(s2_frac * (n - 1))))
    # split segments; keep consistent legend order and ensure S1 stays visible even if short
    segs = [
        ("S1", "#3b82f6", tx[: idx1 + 1], ty[: idx1 + 1]),
        ("S2", "#f59e0b", tx[idx1: idx2 + 1], ty[idx1: idx2 + 1]),
        ("S3", "#10b981", tx[idx2:], ty[idx2:]),
    ]
    fig = go.Figure()
    for name, color, sx, sy in segs:
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=color, width=3), name=name, hoverinfo="skip"))

    smin = float(sim_df["speed_val"].min()) if sim_df["speed_val"].notna().any() else 0
    smax = float(sim_df["speed_val"].max()) if sim_df["speed_val"].notna().any() else 1
    first_frame = sim_df["frame"].min()
    init = sim_df[sim_df["frame"] == first_frame]
    fig.add_trace(
        go.Scatter(
            x=init["x"],
            y=init["y"],
            mode="markers",
            marker=dict(size=12, color=init["speed_val"], colorscale="Turbo", showscale=True, cmin=smin, cmax=smax, colorbar=dict(title="Speed (kph)")),
            text=init["vehicle_number"],
            hovertemplate="Car %{text}<br>t=%{customdata[0]:.2f}s<extra></extra>",
            customdata=np.stack([init["elapsed_s"]], axis=1),
            name="Cars",
            showlegend=False,
        )
    )
    car_trace_idx = len(fig.data) - 1
    frames = []
    for frame_val, group in sim_df.groupby("frame"):
        frames.append(
            go.Frame(
                name=str(frame_val),
                data=[
                    go.Scatter(
                        x=group["x"],
                        y=group["y"],
                        mode="markers",
                        marker=dict(size=12, color=group["speed_val"], colorscale="Turbo", showscale=False, cmin=smin, cmax=smax),
                        text=group["vehicle_number"],
                        hovertemplate="Car %{text}<br>t=%{customdata[0]:.2f}s<extra></extra>",
                        customdata=np.stack([group["elapsed_s"]], axis=1),
                        name="Cars",
                        showlegend=False,
                    )
                ],
                traces=[car_trace_idx],
            )
        )
    fig.frames = frames
    fig.update_layout(
        title="Fastest-lap simulation (raw telemetry driven)",
        xaxis=dict(showgrid=False, zeroline=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, range=[-0.05, 1.05]),
        template="plotly_white",
        legend=dict(x=0.82, y=1, bgcolor="rgba(255,255,255,0.15)"),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ],
    )
    return fig

# ------------------------
# Small Helpers for UI
# ------------------------

def filter_laps_by_duration(df, min_sec=60, max_sec=200):
    """Optionally drop laps outside a duration band to remove obvious outliers/inlaps."""
    if df.empty: return df
    if 'LAP_TIME_SEC' not in df.columns: return df
    return df[(df['LAP_TIME_SEC'] >= min_sec) & (df['LAP_TIME_SEC'] <= max_sec)]

def format_sec(sec):
    """Lightweight mm:ss.s formatter for metrics."""
    if pd.isna(sec): return "—"
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:05.2f}"

def render_data_status(config):
    """Lightweight callout showing which data files are present."""
    lines = []
    end_path = Path(config['endurance_file'])
    lines.append(f"Endurance: {' ✓' if end_path.exists() else ' ! missing'} `{end_path.name}`")
    tele_file = config.get('telemetry_file')
    if tele_file:
        tele_path = Path(tele_file)
        lines.append(f"Telemetry: {' ✓' if tele_path.exists() else ' ! missing'} `{tele_path.name}`")
    ref_file = config.get('telemetry_ref_file')
    if ref_file:
        ref_path = Path(ref_file)
        lines.append(f"Reference lap: {' ✓' if ref_path.exists() else ' ! missing'} `{ref_path.name}`")
    st.caption("Data status: " + " • ".join(lines))

def render_top_kpis(mode, target_row, ref_row):
    """Show a compact KPI row that adapts to single vs compare modes."""
    if target_row is None or target_row.empty: return
    if mode == "Driver vs. Fastest Lap":
        c1, c2 = st.columns(2)
        c1.metric(f"Driver {target_row['NUMBER']} Lap", format_sec(target_row['LAP_TIME_SEC']))
        rank = target_row.get('RANK', np.nan)
        c2.metric("Lap Rank", f"{int(rank):d}" if not pd.isna(rank) else "—")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Driver A ({target_row['NUMBER']})", format_sec(target_row['LAP_TIME_SEC']))
        c2.metric(f"Driver B ({ref_row['NUMBER']})", format_sec(ref_row['LAP_TIME_SEC']))
        if pd.isna(target_row['LAP_TIME_SEC']) or pd.isna(ref_row['LAP_TIME_SEC']):
            delta = "—"
        else:
            delta = f"{target_row['LAP_TIME_SEC'] - ref_row['LAP_TIME_SEC']:+.2f}s"
        c3.metric("Lap Delta (A-B)", delta)

# ------------------------
# Playback Helpers (Map + Plots)
# ------------------------

def load_and_align_telemetry(config, track_name, ref_driver, target_driver, ref_row, target_row, ref_lap_num, target_lap_num):
    """Load telemetry for two drivers, align to common station grid, and store in session_state."""
    ref_target_time = ref_row['LAP_TIME_SEC']
    target_target_time = target_row['LAP_TIME_SEC']

    real_ref_lap, ref_diff = find_closest_telemetry_lap(config['telemetry_file'], ref_driver, ref_target_time, target_lap_num=ref_lap_num)
    real_target_lap, target_diff = find_closest_telemetry_lap(config['telemetry_file'], target_driver, target_target_time, target_lap_num=target_lap_num)

    if real_ref_lap is None:
        st.warning(f"Could not find telemetry for Ref Driver {ref_driver}. Using Endurance Lap {ref_lap_num}.")
        real_ref_lap = ref_lap_num
    if real_target_lap is None:
        st.warning(f"Could not find telemetry for Target Driver {target_driver}. Using Endurance Lap {target_lap_num}.")
        real_target_lap = target_lap_num

    ref_raw = load_telemetry_for_lap(config['telemetry_file'], ref_driver, real_ref_lap)
    target_raw = load_telemetry_for_lap(config['telemetry_file'], target_driver, real_target_lap)
    if ref_raw.empty or target_raw.empty:
        st.error("Telemetry data not found for one or both drivers.")
        return False

    ref_proc = process_telemetry_data(ref_raw)
    target_proc = process_telemetry_data(target_raw)

    centerline_data = track_projection.generate_centerline(ref_proc)
    ref_proj = track_projection.project_to_centerline(ref_proc, centerline_data)
    target_proj = track_projection.project_to_centerline(target_proc, centerline_data)

    max_station = ref_proj['station'].max()
    station_grid = np.arange(0, max_station, 1.0)  # 1m grid

    ref_aligned = track_projection.align_data_to_station(ref_proj, station_grid)
    target_aligned = track_projection.align_data_to_station(target_proj, station_grid)
    ref_aligned['dist'] = ref_aligned['station']
    target_aligned['dist'] = target_aligned['station']

    st.session_state['ref_telemetry'] = ref_aligned
    st.session_state['target_telemetry'] = target_aligned
    st.session_state['telemetry_loaded'] = True
    st.session_state['loaded_drivers'] = (ref_driver, target_driver)
    st.session_state['telemetry_track'] = track_name
    st.session_state['telemetry_max_dist'] = max_station
    return True


def plot_playback_map(ref_df, target_df, ref_driver, target_driver, dist_value):
    """Simple map with moving dots at the current distance."""
    if ref_df.empty or target_df.empty:
        return plt.figure()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw base track with higher contrast
    ax.plot(ref_df['x'], ref_df['y'], color='white', linewidth=3, alpha=0.65, solid_capstyle='round')
    ax.plot(ref_df['x'], ref_df['y'], color='gray', linewidth=6, alpha=0.25, solid_capstyle='round')
    # Draw dots
    def nearest_point(df, d):
        idx = (df['dist'] - d).abs().idxmin()
        return df.loc[idx, ['x', 'y']]
    rx, ry = nearest_point(ref_df, dist_value)
    tx, ty = nearest_point(target_df, dist_value)
    ax.scatter(rx, ry, color='deepskyblue', edgecolor='white', linewidth=1.2, s=70, label=f"{ref_driver}")
    ax.scatter(tx, ty, color='orangered', edgecolor='white', linewidth=1.2, s=70, label=f"{target_driver}")

    buffer = 50
    ax.set_xlim(min(ref_df['x'].min(), target_df['x'].min()) - buffer, max(ref_df['x'].max(), target_df['x'].max()) + buffer)
    ax.set_ylim(min(ref_df['y'].min(), target_df['y'].min()) - buffer, max(ref_df['y'].max(), target_df['y'].max()) + buffer)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)
    ax.set_title("Playback: Moving Dots", color='white', fontsize=11)
    return fig


def plot_channel(ref_df, target_df, dist_value, column, label, y_label):
    """Overlay a telemetry channel with a cursor at current distance."""
    if column not in ref_df.columns or column not in target_df.columns:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(0.5, 0.5, f"{label} not available", ha='center', va='center')
        ax.axis('off')
        return fig
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(ref_df['dist'], ref_df[column], color='deepskyblue', label='Ref')
    ax.plot(target_df['dist'], target_df[column], color='orangered', label='Target')
    ax.axvline(dist_value, color='green', linestyle='--', linewidth=2)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def render_playback_section(analysis_mode, track_name, config, ref_driver, target_driver, ref_row, target_row, ref_lap_num, target_lap_num):
    """Render playback controls and synchronized map/plots (Barber telemetry only for now)."""
    if not config.get('telemetry_file'):
        st.info("Playback not available for this track yet.")
        return
    if track_name != 'Barber':
        st.info("Playback demo currently wired for Barber telemetry only.")
        return

    st.caption("Load telemetry to drive moving dots and synced plots.")

    needs_load = not st.session_state.get('telemetry_loaded') or st.session_state.get('telemetry_track') != track_name
    drivers_mismatch = st.session_state.get('loaded_drivers') != (ref_driver, target_driver)
    if needs_load or drivers_mismatch:
        st.info("Load telemetry for the current driver pair to enable playback.")
        if st.button("Load telemetry for playback", type="primary"):
            with st.spinner("Loading and aligning telemetry..."):
                ok = load_and_align_telemetry(config, track_name, ref_driver, target_driver, ref_row, target_row, ref_lap_num, target_lap_num)
                if not ok:
                    return
        else:
            return  # Wait for user to load

    if not st.session_state.get('telemetry_loaded'):
        return

    ref_data = st.session_state['ref_telemetry']
    target_data = st.session_state['target_telemetry']
    max_dist = st.session_state.get('telemetry_max_dist', max(ref_data['dist'].max(), target_data['dist'].max()))
    # Playback control: manual slider only
    default_pos = float(max_dist * 0.1)
    dist_value = st.slider("Lap position (m)", 0.0, float(max_dist), float(default_pos), step=1.0, key="playback_dist")

    map_col, _ = st.columns([1.3, 1.7])
    with map_col:
        st.pyplot(plot_playback_map(ref_data, target_data, ref_driver, target_driver, dist_value))
    st.caption(f"Telemetry loaded: {st.session_state.get('loaded_drivers')} | Track: {st.session_state.get('telemetry_track')}")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_channel(ref_data, target_data, dist_value, 'aps', 'Throttle (%)', 'Throttle (%)'))
        st.pyplot(plot_channel(ref_data, target_data, dist_value, 'speed', 'Speed', 'Speed (km/h)'))
    with c2:
        st.pyplot(plot_channel(ref_data, target_data, dist_value, 'pbrake_f', 'Brake', 'Brake (bar)'))
        if 'gear' in ref_data.columns and 'gear' in target_data.columns:
            st.pyplot(plot_channel(ref_data, target_data, dist_value, 'gear', 'Gear', 'Gear'))
        else:
            st.empty()

def render_driver_profile_section(track_name, config):
    """Driver style profiling via radar and quick metrics (telemetry tracks only)."""
    if not config.get('telemetry_file'):
        st.info("Driver profiling needs telemetry; not available for this track yet.")
        return
    st.header(f"Driver Profile: {track_name}")
    tele_path = Path(config['telemetry_file'])
    if not tele_path.exists():
        st.warning(f"Telemetry file missing: {tele_path.name}")
        return
    # Cap Barber to 28 to include laps numbered up to 29 while excluding obvious extras
    lap_cap = 28 if track_name == "Barber" else None
    tele_wide = load_telemetry_wide(config['telemetry_file'], lap_cap=lap_cap)
    if tele_wide.empty:
        st.info("No telemetry loaded for profiling (empty file or bad schema).")
        return
    lap_features, driver_profile = compute_lap_features(tele_wide)
    if driver_profile.empty:
        st.info("Could not compute driver stats from telemetry.")
        return
    radar_scaled, radar_features, radar_labels = prep_radar(driver_profile)
    if radar_scaled.empty:
        st.info("No radar metrics available.")
        return

    drivers = sorted(radar_scaled.index.tolist())
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        driver_a = st.selectbox("Driver A", drivers, key="driver_profile_a")
    with c_sel2:
        driver_b = st.selectbox("Driver B (optional compare)", ["(none)"] + drivers, key="driver_profile_b")
        driver_b = driver_b if driver_b != "(none)" else None

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        laps_a = int(driver_profile.loc[driver_a, "n_laps"]) if driver_a in driver_profile.index else 0
        st.plotly_chart(build_radar_plot(radar_scaled, radar_features, radar_labels, driver_a, laps_a, color="deepskyblue", title_prefix=""), use_container_width=True)
    with col_r2:
        if driver_b:
            laps_b = int(driver_profile.loc[driver_b, "n_laps"]) if driver_b in driver_profile.index else 0
            st.plotly_chart(build_radar_plot(radar_scaled, radar_features, radar_labels, driver_b, laps_b, color="orangered", title_prefix=""), use_container_width=True)
        else:
            st.info("Pick a second driver to compare.")

    # Quick stats table
    display_cols = ["n_laps", "lap_duration", "speed_max", "throttle_mean", "pct_full_throttle", "pct_brake", "lap_consistency"]
    stats = driver_profile[display_cols].rename(columns={
        "n_laps": "Laps",
        "lap_duration": "Avg Lap (s)",
        "speed_max": "Max Speed",
        "throttle_mean": "Avg Throttle",
        "pct_full_throttle": "Full Throttle %",
        "pct_brake": "Brake %",
        "lap_consistency": "Consistency",
    })
    st.dataframe(stats.style.format({
        "Avg Lap (s)": "{:.2f}",
        "Max Speed": "{:.1f}",
        "Avg Throttle": "{:.1f}",
        "Full Throttle %": "{:.0%}",
        "Brake %": "{:.0%}",
        "Consistency": "{:.3f}",
    }), use_container_width=True, height=260)

# COTA Detailed Points (Hardcoded)
COTA_FULL_POINTS = {
    'Start': (98, 466, 115, 460),
    'T1_Start': (290, 660, 303, 644), 'T1_Apex': (303, 644, 318, 662), 'T1_Exit': (303, 644, 322, 641),
    'T2_Start': (290, 576, 307, 576), 'T2_Apex': (301, 549, 314, 560), 'T2_Exit': (319, 527, 326, 542),
    'T3_Start': (384, 499, 395, 512), 'T3_Apex': (396, 494, 410, 508), 'T3_Exit': (404, 481, 421, 491),
    'T4_Start': (409, 467, 427, 477), 'T4_Apex': (424, 459, 436, 469), 'T4_Exit': (441, 449, 449, 464),
    'T5_Start': (441, 449, 449, 464), 'T5_Apex': (457, 442, 470, 454), 'T5_Exit': (463, 430, 481, 437),
    'T6_Start': (470, 406, 484, 418), 'T6_Apex': (517, 337, 521, 397), 'T6_Exit': (544, 384, 532, 401),
    'T7_Start': (566, 400, 558, 412), 'T7_Apex': (582, 410, 582, 428), 'T7_Exit': (596, 401, 608, 415),
    'T8_Start': (614, 390, 628, 401), 'T8_Apex': (643, 377, 642, 392), 'T8_Exit': (667, 392, 652, 402),
    'T9_Start': (667, 392, 652, 402), 'T9_Apex': (670, 400, 660, 413), 'T9_Exit': (681, 403, 685, 420),
    'T10_Start': (740, 398, 740, 416), 'T10_Apex': (752, 400, 760, 415), 'T10_Exit': (762, 389, 777, 400),
    'T11_Start': (809, 310, 830, 318), 'T11_Apex': (815, 303, 836, 294), 'T11_Exit': (807, 306, 805, 287),
    'T12_Start': (341, 334, 341, 317), 'T12_Apex': (327, 335, 308, 322), 'T12_Exit': (333, 347, 315, 354),
    'T13_Start': (357, 399, 337, 399), 'T13_Apex': (359, 421, 342, 411), 'T13_Exit': (332, 428, 334, 411),
    'T14_Start': (319, 423, 322, 407), 'T14_Apex': (303, 416, 316, 404), 'T14_Exit': (297, 399, 314, 396),
    'T15_Start': (285, 375, 303, 370), 'T15_Apex': (281, 367, 293, 355), 'T15_Exit': (277, 368, 275, 350),
    'T16_Start': (277, 368, 275, 350), 'T16_Apex': (269, 366, 253, 351), 'T16_Exit': (274, 379, 256, 382),
    'T17_Start': (292, 427, 280, 446), 'T17_Apex': (280, 490, 266, 475), 'T17_Exit': (256, 497, 250, 481),
    'T18_Start': (256, 497, 250, 481), 'T18_Apex': (197, 495, 206, 479), 'T18_Exit': (166, 461, 182, 449),
    'T19_Start': (143, 417, 157, 406), 'T19_Apex': (131, 400, 136, 384), 'T19_Exit': (115, 403, 106, 388),
    'T20_Start': (95, 405, 80, 390), 'T20_Apex': (80, 420, 60, 400), 'T20_Exit': (75, 440, 95, 425),
    'Finish': (98, 466, 115, 460)
}

# ------------------------
# Data Loading Functions
# ------------------------

def time_to_sec(t):
    if isinstance(t, str):
        if ':' in t:
            try:
                parts = t.split(':')
                if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
                if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            except: return np.nan
        else:
            try: return float(t)
            except: return np.nan
    if pd.isna(t): return np.nan 
    return float(t)

@st.cache_data
def load_race_data(file_path, micro_sectors):
    """Loads endurance data for sector analysis."""
    try:
        endurance = pd.read_csv(file_path, sep=';')
        endurance.columns = endurance.columns.str.strip()
        if 'LAP_TIME' not in endurance.columns and len(endurance.columns) == 1:
             endurance = pd.read_csv(file_path, sep=',')
             endurance.columns = endurance.columns.str.strip()

        endurance['LAP_TIME_SEC'] = endurance['LAP_TIME'].apply(time_to_sec)
        for _, col in micro_sectors:
            if col in endurance.columns:
                endurance[col + '_SEC'] = endurance[col].apply(time_to_sec)
            else:
                endurance[col + '_SEC'] = np.nan 
            
        valid_laps = endurance.dropna(subset=['LAP_TIME_SEC'])
        if not valid_laps.empty:
            ranked_laps = valid_laps.groupby('NUMBER')['LAP_TIME_SEC'].min().rank(method='min', ascending=True).sort_values()
            endurance['RANK'] = endurance['NUMBER'].map(ranked_laps)
        else:
            endurance['RANK'] = np.nan
            
        return endurance
    except Exception as e:
        st.error(f"Error loading race data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_vehicle_id_map(file_path):
    """Scans the telemetry file to map Driver Number -> Vehicle ID."""
    vehicle_map = {}
    try:
        unique_ids = set()
        for chunk in pd.read_csv(file_path, usecols=['vehicle_id'], chunksize=1000000):
            unique_ids.update(chunk['vehicle_id'].dropna().unique())
            
        for vid in unique_ids:
            parts = vid.replace('GR86-', '').split('-')
            for p in parts:
                if p.isdigit():
                    vehicle_map[int(p)] = vid
        return vehicle_map
    except Exception as e:
        st.error(f"Error mapping vehicle IDs: {e}")
        return {}

@st.cache_data
def find_closest_telemetry_lap(file_path, vehicle_number, target_time, target_lap_num=None, tolerance=2.0, min_duration=80):
    """Finds the telemetry lap number that best matches the endurance lap time. Prioritizes exact lap match."""
    v_map = get_vehicle_id_map(file_path)
    vehicle_id_str = v_map.get(int(vehicle_number))
    if not vehicle_id_str:
        vehicle_id_str = f"GR86-{int(vehicle_number):03d}-000"
        
    chunk_size = 1000000
    lap_durations = {}
    
    try:
        for chunk in pd.read_csv(file_path, usecols=['vehicle_id', 'lap', 'meta_time'], chunksize=chunk_size):
            driver_chunk = chunk[chunk['vehicle_id'] == vehicle_id_str]
            if driver_chunk.empty: continue
            
            driver_chunk = driver_chunk.copy()
            driver_chunk['meta_time'] = pd.to_datetime(driver_chunk['meta_time'])
            
            for lap, group in driver_chunk.groupby('lap'):
                if lap not in lap_durations:
                    lap_durations[lap] = {'min': group['meta_time'].min(), 'max': group['meta_time'].max()}
                else:
                    lap_durations[lap]['min'] = min(lap_durations[lap]['min'], group['meta_time'].min())
                    lap_durations[lap]['max'] = max(lap_durations[lap]['max'], group['meta_time'].max())
        
        # 1. Check if the target_lap_num exists and matches the time
        if target_lap_num is not None and target_lap_num in lap_durations:
            times = lap_durations[target_lap_num]
            duration = (times['max'] - times['min']).total_seconds()
            if duration > min_duration:
                diff = abs(duration - target_time)
                if diff <= tolerance:
                    return target_lap_num, diff
                    
        # 2. If not, scan for best match
        best_lap = None
        min_diff = float('inf')
        
        for lap, times in lap_durations.items():
            duration = (times['max'] - times['min']).total_seconds()
            
            # Filter out partial laps
            if duration < min_duration: continue
            
            diff = abs(duration - target_time)
            
            if diff < min_diff:
                min_diff = diff
                best_lap = lap
                
        if best_lap is not None and min_diff <= tolerance:
            return best_lap, min_diff
        else:
            # Fallback: If no close match, try to find the absolute closest even if outside tolerance
            if best_lap is not None:
                 return best_lap, min_diff
            return None, None
            
    except Exception as e:
        st.error(f"Error scanning telemetry laps: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def load_telemetry_wide(file_path, lap_cap=None):
    """Load raw telemetry and pivot to wide format (Barber/COTA schema)."""
    try:
        tele = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading telemetry: {e}")
        return pd.DataFrame()

    # Normalize vehicle numbers from vehicle_id suffix (handles 2 vs 02 consistently)
    def extract_num(v):
        if pd.isna(v):
            return ""
        return str(v).rsplit("-", 1)[-1]

    tele["vehicle_number"] = tele["vehicle_id"].apply(extract_num)
    tele["vehicle_number"] = tele["vehicle_number"].str.zfill(2)
    tele = tele[tele["vehicle_number"] != "000"]
    tele["meta_time"] = pd.to_datetime(tele["meta_time"])
    tele["timestamp"] = pd.to_datetime(tele["timestamp"])
    tele = tele.sort_values(["vehicle_number", "meta_time"])

    tele_wide = (
        tele.pivot_table(
            index=["vehicle_number", "lap", "meta_time", "timestamp"],
            columns="telemetry_name",
            values="telemetry_value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["vehicle_number", "lap", "meta_time"])
    )
    if lap_cap is not None:
        tele_wide["lap"] = pd.to_numeric(tele_wide["lap"], errors="coerce")
        tele_wide = tele_wide[tele_wide["lap"] <= lap_cap]
    return tele_wide


def compute_lap_features(tele_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-lap statistics for radar plots and best-lap selection."""
    if tele_wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["vehicle_number", "lap"]
    # Handle throttle column differences (COTA uses 'ath', Barber uses 'aps')
    throttle_col = None
    if "ath" in tele_wide.columns:
        throttle_col = "ath"
    elif "aps" in tele_wide.columns:
        throttle_col = "aps"

    def pct_full_throttle(series):
        if throttle_col is None:
            return np.nan
        vals = pd.to_numeric(series, errors="coerce")
        return (vals > 90).mean()

    def throttle_mean(series):
        if throttle_col is None:
            return np.nan
        vals = pd.to_numeric(series, errors="coerce")
        return vals.mean()

    agg_map = dict(
        lap_start=("meta_time", "min"),
        lap_end=("meta_time", "max"),
        lap_duration=("meta_time", lambda x: (x.max() - x.min()).total_seconds()),
        speed_mean=("speed", "mean"),
        speed_max=("speed", "max"),
        brake_f_mean=("pbrake_f", "mean"),
        pct_brake=("pbrake_f", lambda x: (pd.to_numeric(x, errors="coerce") > 5).mean()),
        accy_max=("accy_can", "max"),
        steer_std=("Steering_Angle", "std"),
        max_brake_g=("accx_can", "min"),
    )
    if throttle_col:
        agg_map["throttle_mean"] = (throttle_col, lambda x: pd.to_numeric(x, errors="coerce").mean())
        agg_map["pct_full_throttle"] = (throttle_col, lambda x: (pd.to_numeric(x, errors="coerce") > 90).mean())
    else:
        agg_map["throttle_mean"] = ("speed", lambda x: np.nan)
        agg_map["pct_full_throttle"] = ("speed", lambda x: np.nan)

    lap_features = tele_wide.groupby(group_cols).agg(**agg_map)

    lap_time_std = lap_features.groupby("vehicle_number")["lap_duration"].std()
    driver_profile = lap_features.groupby("vehicle_number").mean(numeric_only=True)
    driver_profile["n_laps"] = lap_features.groupby("vehicle_number").size()
    driver_profile["lap_time_std"] = lap_time_std
    driver_profile["lap_consistency"] = 1 / (1 + driver_profile["lap_time_std"])

    return lap_features.reset_index(), driver_profile


def prep_radar(driver_profile: pd.DataFrame) -> tuple[pd.DataFrame, list, dict]:
    """Scale radar features to 0-1 range and provide readable labels."""
    if driver_profile.empty:
        return pd.DataFrame(), [], {}
    radar_df = driver_profile.copy()
    radar_df["max_brake_g_pos"] = -radar_df["max_brake_g"]
    features = [
        "pct_full_throttle",
        "accy_max",
        "steer_std",
        "pct_brake",
        "max_brake_g_pos",
        "lap_consistency",
    ]
    labels = {
        "pct_full_throttle": "Full Throttle %",
        "accy_max": "Max Lateral G",
        "steer_std": "Steering Variability",
        "pct_brake": "Brake Time %",
        "max_brake_g_pos": "Max Brake G",
        "lap_consistency": "Lap Consistency",
    }
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(radar_df[features].fillna(0))
    radar_scaled = pd.DataFrame(scaled, index=radar_df.index, columns=features)
    return radar_scaled, features, labels


def build_radar_plot(
    radar_scaled: pd.DataFrame,
    features: list,
    labels: dict,
    driver: str,
    n_laps: int,
    color: str = "purple",
    title_prefix: str = "",
) -> go.Figure:
    values = radar_scaled.loc[driver, features].tolist()
    values += values[:1]  # close loop
    angles = np.linspace(0, 2 * np.pi, len(values))
    ticktext = [labels.get(f, f) for f in features]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=np.degrees(angles),
            fill="toself",
            name=driver,
            line=dict(color=color),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickmode="array",
                tickvals=np.degrees(angles[:-1]),
                ticktext=ticktext,
                tickfont=dict(size=9),
            ),
        ),
        showlegend=False,
        title=dict(
            text=f"{title_prefix}Driver {driver} – Radar ({n_laps} laps)",
            pad=dict(b=8),
        ),
        margin=dict(t=40, b=20, l=10, r=10),
        height=320,
    )
    return fig

@st.cache_data(show_spinner=True)
def load_telemetry_for_lap(file_path, vehicle_number, lap_number):
    """Loads telemetry data for a specific vehicle and lap."""
    chunk_size = 500000 
    relevant_chunks = []
    
    target_sensors = [
        'aps', 'pbrake_f', 'pbrake_r', 'Steering_Angle', 'speed', 'nmot', 
        'accx_can', 'accy_can', 'gear', 'Laptrigger_lapdist_dls',
        'VBOX_Lat_Min', 'VBOX_Long_Minutes'
    ]
    
    v_map = get_vehicle_id_map(file_path)
    vehicle_id_str = v_map.get(int(vehicle_number))
    
    if not vehicle_id_str:
        vehicle_id_str = f"GR86-{int(vehicle_number):03d}-000"
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if 'vehicle_id' in chunk.columns:
                mask = (chunk['vehicle_id'] == vehicle_id_str) & (chunk['lap'] == lap_number)
                filtered_chunk = chunk[mask]
                if not filtered_chunk.empty:
                    sensor_mask = filtered_chunk['telemetry_name'].isin(target_sensors)
                    relevant_chunks.append(filtered_chunk[sensor_mask])
                
        if relevant_chunks:
            return pd.concat(relevant_chunks)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading telemetry file: {e}")
        return pd.DataFrame()

def process_telemetry_data(df):
    """Pivots and processes telemetry data."""
    if df.empty: return pd.DataFrame()
    
    df = df.drop_duplicates(subset=['meta_time', 'telemetry_name'])
    df = df.copy()
    df['meta_time'] = pd.to_datetime(df['meta_time'])
    
    pivot_df = df.pivot(index='meta_time', columns='telemetry_name', values='telemetry_value')
    
    for col in pivot_df.columns:
        pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce')
        
    pivot_df.sort_index(inplace=True)
    pivot_df = pivot_df.interpolate(method='time').dropna(how='all')
    
    # Add relative time in seconds for interpolation
    if not pivot_df.empty:
        pivot_df['time_sec'] = (pivot_df.index - pivot_df.index[0]).total_seconds()
    
    # Convert GPS to XY
    if 'VBOX_Lat_Min' in pivot_df.columns and 'VBOX_Long_Minutes' in pivot_df.columns:
        lat0 = pivot_df['VBOX_Lat_Min'].iloc[0]
        lon0 = pivot_df['VBOX_Long_Minutes'].iloc[0]
        R = 6371000
        pivot_df['x'] = (pivot_df['VBOX_Long_Minutes'] - lon0) * (np.pi / 180) * R * np.cos(lat0 * np.pi / 180)
        pivot_df['y'] = (pivot_df['VBOX_Lat_Min'] - lat0) * (np.pi / 180) * R

    # Distance alignment
    if 'Laptrigger_lapdist_dls' in pivot_df.columns:
        pivot_df['dist'] = pivot_df['Laptrigger_lapdist_dls']
        pivot_df.sort_values('dist', inplace=True)
        
        if not pivot_df['dist'].empty:
            # Zero time at the start of the lap (min distance)
            # Find the row with the minimum distance (closest to start line)
            start_idx = pivot_df['dist'].idxmin()
            start_time = pivot_df.loc[start_idx, 'time_sec']
            pivot_df['time_sec'] = pivot_df['time_sec'] - start_time
            
            max_dist = pivot_df['dist'].max()
            grid_dist = np.arange(0, max_dist, 1.0)
            pivot_df = pivot_df.drop_duplicates(subset=['dist'])
            
            resampled_data = {}
            for col in pivot_df.columns:
                if col != 'dist':
                    resampled_data[col] = np.interp(grid_dist, pivot_df['dist'], pivot_df[col])
            
            resampled_df = pd.DataFrame(resampled_data)
            resampled_df['dist'] = grid_dist
            return resampled_df
            
    return pivot_df

# ------------------------
# Track Generation
# ------------------------

@st.cache_data
def load_cota_track():
    """Generates COTA track points from hardcoded pixels."""
    centerline_points = []
    key_to_index = {}
    new_point_count = 0
    
    def interpolate_point(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    smoothing_targets = ['T7_Exit', 'T11_Exit', 'T19_Exit']

    for index, (key, (ix, iy, ox, oy)) in enumerate(COTA_FULL_POINTS.items()):
        center_x = (ix + ox) / 2
        center_y = (iy + oy) / 2
        current_point = (center_x, center_y)

        centerline_points.append(current_point)
        key_to_index[key] = index + new_point_count

        if key in smoothing_targets and len(centerline_points) > 1:
            p_prev = centerline_points[-2]
            p_curr = current_point
            new_smooth_point = interpolate_point(p_prev, p_curr)
            centerline_points.insert(len(centerline_points) - 1, new_smooth_point)
            new_point_count += 1
            key_to_index[key] = index + new_point_count
            
    key_to_index['Finish'] = len(centerline_points) - 1
    
    # Define Sector Segments
    sector_indices = [
        (key_to_index['Start'], key_to_index['T1_Exit']),
        (key_to_index['T1_Exit'], key_to_index['T7_Exit']),
        (key_to_index['T7_Exit'], key_to_index['T11_Exit']),
        (key_to_index['T11_Exit'], key_to_index['T15_Exit']),
        (key_to_index['T15_Exit'], key_to_index['T19_Exit']),
        (key_to_index['T19_Exit'], key_to_index['Finish'])
    ]
    
    return centerline_points, sector_indices

@st.cache_data
def load_barber_track(telemetry_file):
    """Generates Barber track points from GPS telemetry."""
    try:
        df = pd.read_csv(telemetry_file)
        lat_rows = df[df['telemetry_name'] == 'VBOX_Lat_Min'].copy()
        lon_rows = df[df['telemetry_name'] == 'VBOX_Long_Minutes'].copy()
        
        lat_rows.rename(columns={'telemetry_value': 'lat'}, inplace=True)
        lon_rows.rename(columns={'telemetry_value': 'lon'}, inplace=True)
        
        lat_rows['lat'] = pd.to_numeric(lat_rows['lat'], errors='coerce')
        lon_rows['lon'] = pd.to_numeric(lon_rows['lon'], errors='coerce')
        
        lat_rows = lat_rows.drop_duplicates(subset=['meta_time'])
        lon_rows = lon_rows.drop_duplicates(subset=['meta_time'])
        
        track_df = pd.merge(lat_rows[['meta_time', 'lat']], lon_rows[['meta_time', 'lon']], on='meta_time', how='inner')
        track_df.sort_values('meta_time', inplace=True)
        
        lat0 = track_df['lat'].iloc[0]
        lon0 = track_df['lon'].iloc[0]
        R = 6371000 
        
        track_df['x'] = (track_df['lon'] - lon0) * (np.pi / 180) * R * np.cos(lat0 * np.pi / 180)
        track_df['y'] = (track_df['lat'] - lat0) * (np.pi / 180) * R
        
        x_raw = track_df['x'].values
        y_raw = track_df['y'].values
        
        dist = np.cumsum(np.sqrt(np.diff(x_raw, prepend=x_raw[0])**2 + np.diff(y_raw, prepend=y_raw[0])**2))
        dist_steps = np.arange(0, dist[-1], 1.0)
        x_interp = np.interp(dist_steps, dist, x_raw)
        y_interp = np.interp(dist_steps, dist, y_raw)
        
        window_length = 51
        x_smooth = savgol_filter(x_interp, window_length, 3)
        y_smooth = savgol_filter(y_interp, window_length, 3)
        
        centerline_points = list(zip(x_smooth, y_smooth))
        
        total_points = len(centerline_points)
        seg_len = int(total_points / 6)
        sector_indices = []
        for i in range(6):
            start = i * seg_len
            end = (i + 1) * seg_len if i < 5 else total_points - 1
            sector_indices.append((start, end))
            
        return centerline_points, sector_indices
    except Exception as e:
        st.error(f"Error loading Barber track: {e}")
        return [], []

# ------------------------
# Visualization Functions
# ------------------------

def plot_heatmap(df, metric, title, cmap='plasma', dynamic_scaling=False, annotations=None):
    """Plots the track map colored by a specific metric."""
    if df.empty or 'x' not in df.columns or 'y' not in df.columns:
        return plt.figure()

    x = df['x'].values
    y = df['y'].values
    z = df[metric].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')

    if dynamic_scaling:
        vmin, vmax = np.percentile(z, [2, 98])
    else:
        if 'aps' in title.lower(): vmin, vmax = 0, 100
        elif 'pbrake' in title.lower(): vmin, vmax = 0, 50
        elif 'speed' in title.lower(): vmin, vmax = 60, 200
        else: vmin, vmax = np.percentile(z, [2, 98])
        
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(5)
    ax.add_collection(lc)
    
    ax.set_xlim(x.min() - 50, x.max() + 50)
    ax.set_ylim(y.min() - 50, y.max() + 50)
    
    if annotations:
        for (ax_x, ax_y, label) in annotations:
            ax.text(ax_x, ax_y, label, color='white', fontsize=12, ha='center', va='center', fontweight='bold', zorder=10,
                    bbox=dict(facecolor='black', edgecolor='none', alpha=0.5, pad=2))
    
    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label(title, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(f"Track Heatmap: {title}", color='white', fontsize=14)
    return fig

def plot_speed_delta_heatmap(ref_df, target_df, ref_driver, target_driver, sensitivity=20, annotations=None):
    """Plots the track map colored by Speed Delta (Target - Ref)."""
    if ref_df.empty or target_df.empty: return plt.figure()
    
    # Merge on distance (assuming 1m grid from process_telemetry_data)
    # We need to ensure both have 'dist' and 'speed'
    if 'dist' not in ref_df.columns or 'dist' not in target_df.columns:
        return plt.figure()
        
    # Merge with tolerance or exact match since we resampled to integers
    merged = pd.merge_asof(target_df.sort_values('dist'), ref_df.sort_values('dist'), on='dist', suffixes=('_target', '_ref'), direction='nearest')
    
    merged['speed_delta'] = merged['speed_target'] - merged['speed_ref']
    
    x = merged['x_target'].values
    y = merged['y_target'].values
    z = merged['speed_delta'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Diverging colormap: Red (Slower) -> White (Equal) -> Green (Faster)
    norm = plt.Normalize(-sensitivity, sensitivity) 
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
    lc.set_array(z)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)
    
    ax.set_xlim(x.min() - 50, x.max() + 50)
    ax.set_ylim(y.min() - 50, y.max() + 50)
    
    if annotations:
        for (ax_x, ax_y, label) in annotations:
            ax.text(ax_x, ax_y, label, color='white', fontsize=12, ha='center', va='center', fontweight='bold', zorder=10,
                    bbox=dict(facecolor='black', edgecolor='none', alpha=0.5, pad=2))
    
    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label(f"Speed Delta (km/h)\n{target_driver} vs {ref_driver}", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(f"Speed Delta: {target_driver} vs {ref_driver}", color='white', fontsize=14)
    return fig

def plot_dual_speed_map(ref_df, target_df, ref_driver, target_driver):
    """Plots both drivers' racing lines colored by their absolute speed."""
    if ref_df.empty or target_df.empty: return plt.figure()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Common Speed Normalization
    all_speeds = np.concatenate([ref_df['speed'], target_df['speed']])
    vmin, vmax = np.percentile(all_speeds, [5, 95])
    norm = plt.Normalize(vmin, vmax)
    
    # Plot Reference (Blues) - GHOST LINE (Wide, Transparent)
    points_ref = np.array([ref_df['x'], ref_df['y']]).T.reshape(-1, 1, 2)
    segments_ref = np.concatenate([points_ref[:-1], points_ref[1:]], axis=1)
    lc_ref = LineCollection(segments_ref, cmap='Blues', norm=norm, alpha=0.4) # Lower alpha
    lc_ref.set_array(ref_df['speed'])
    lc_ref.set_linewidth(5.0) # Much wider to act as background
    ax.add_collection(lc_ref)
    
    # Plot Target (Reds) - SHARP LINE (Thin, Opaque)
    points_target = np.array([target_df['x'], target_df['y']]).T.reshape(-1, 1, 2)
    segments_target = np.concatenate([points_target[:-1], points_target[1:]], axis=1)
    lc_target = LineCollection(segments_target, cmap='Reds', norm=norm, alpha=1.0) # Full opacity
    lc_target.set_array(target_df['speed'])
    lc_target.set_linewidth(1.5) # Thin and sharp
    ax.add_collection(lc_target)
    
    # Add a legend proxy
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=plt.cm.Blues(0.6), lw=5, alpha=0.4),
                    Line2D([0], [0], color=plt.cm.Reds(0.6), lw=1.5)]
    ax.legend(custom_lines, [f'{ref_driver} (Ref - Blue Ghost)', f'{target_driver} (Target - Red Sharp)'], 
              facecolor='black', edgecolor='white', labelcolor='white')
    
    ax.set_xlim(ref_df['x'].min() - 50, ref_df['x'].max() + 50)
    ax.set_ylim(ref_df['y'].min() - 50, ref_df['y'].max() + 50)
    
    # Dual Colorbars? Or just one explaining intensity?
    # Let's add one for "Speed Intensity" using a neutral map or just explain it.
    # Actually, simpler to just have the legend explain the colors.
    # But we can add a small colorbar for one of them to show the scale.
    cbar = fig.colorbar(lc_target, ax=ax, pad=0.02)
    cbar.set_label("Speed Intensity (Dark=Slow, Bright=Fast)", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(f"Racing Line: {target_driver} (Red) vs {ref_driver} (Blue)", color='white', fontsize=14)
    return fig

def analyze_corner_performance(ref_df, target_df, start_dist, end_dist):
    """Calculates detailed performance metrics for a specific corner."""
    metrics = {}
    
    # Filter data for the corner
    ref_corner = ref_df[(ref_df['dist'] >= start_dist) & (ref_df['dist'] <= end_dist)]
    target_corner = target_df[(target_df['dist'] >= start_dist) & (target_df['dist'] <= end_dist)]
    
    if ref_corner.empty or target_corner.empty: return None
    
    # --- Absolute Apex Calculation ---
    # Refine: Search only in the middle 60% of the turn to avoid entry/exit noise
    turn_len = len(ref_corner)
    search_start = int(turn_len * 0.20)
    search_end = int(turn_len * 0.80)
    
    if search_end > search_start:
        ref_search = ref_corner.iloc[search_start:search_end]
    else:
        ref_search = ref_corner # Fallback to full turn if too short
        
    # Calculate curvature for Ref (Smoothed)
    # We use a larger window for smoothing to find the "Macro" apex
    from scipy.signal import savgol_filter
    try:
        # Smooth X and Y
        sx = savgol_filter(ref_search['x'], window_length=min(15, len(ref_search)), polyorder=3)
        sy = savgol_filter(ref_search['y'], window_length=min(15, len(ref_search)), polyorder=3)
        
        dx = np.gradient(sx)
        dy = np.gradient(sy)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula with epsilon to avoid div by zero
        curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)
        
        # Find index of max curvature relative to the search window
        local_apex_idx = np.argmax(curvature)
        
        # Map back to the original dataframe
        apex_row = ref_search.iloc[local_apex_idx]
        apex_x = apex_row['x']
        apex_y = apex_row['y']
        
    except:
        # Fallback: Geometric Center of the turn
        mid_idx = len(ref_corner) // 2
        apex_x = ref_corner.iloc[mid_idx]['x']
        apex_y = ref_corner.iloc[mid_idx]['y']

    metrics['apex_x'] = apex_x
    metrics['apex_y'] = apex_y
    
    # Calculate Distance to Apex for both drivers
    # Ref distance to Apex Point
    ref_dists = np.sqrt((ref_corner['x'] - apex_x)**2 + (ref_corner['y'] - apex_y)**2)
    metrics['ref_apex_dist'] = ref_dists.min()
    
    # Target distance to Apex Point
    target_dists = np.sqrt((target_corner['x'] - apex_x)**2 + (target_corner['y'] - apex_y)**2)
    metrics['target_apex_dist'] = target_dists.min()
    
    # Speed at Apex (Speed when closest to Apex)
    metrics['ref_speed_at_apex'] = ref_corner.iloc[ref_dists.argmin()]['speed']
    metrics['target_speed_at_apex'] = target_corner.iloc[target_dists.argmin()]['speed']

    # 1. Min Speed (Apex Speed)
    metrics['ref_min_speed'] = ref_corner['speed'].min()
    metrics['target_min_speed'] = target_corner['speed'].min()
    
    # 2. Entry Speed (Speed at start of segment)
    metrics['ref_entry_speed'] = ref_corner['speed'].iloc[0]
    metrics['target_entry_speed'] = target_corner['speed'].iloc[0]
    
    # 3. Exit Speed (Speed at end of segment)
    metrics['ref_exit_speed'] = ref_corner['speed'].iloc[-1]
    metrics['target_exit_speed'] = target_corner['speed'].iloc[-1]
    
    # 4. Coasting Distance (Throttle < 5% and Brake < 1 bar)
    if 'aps' in ref_corner.columns and 'pbrake_f' in ref_corner.columns:
        ref_coast = (ref_corner['aps'] < 5) & (ref_corner['pbrake_f'] < 1)
        metrics['ref_coast_dist'] = ref_coast.sum() # Assuming 1m grid
    else: metrics['ref_coast_dist'] = 0
    
    if 'aps' in target_corner.columns and 'pbrake_f' in target_corner.columns:
        target_coast = (target_corner['aps'] < 5) & (target_corner['pbrake_f'] < 1)
        metrics['target_coast_dist'] = target_coast.sum()
    else: metrics['target_coast_dist'] = 0
    
    # 5. Trail Braking (Distance from Entry to Brake Release < 1 bar)
    # We look for the LAST point where brake > 1 bar
    if 'pbrake_f' in ref_corner.columns:
        braking_mask = ref_corner['pbrake_f'] > 1
        if braking_mask.any():
            last_brake_idx = braking_mask[::-1].idxmax() # Find last True
            # Calculate distance from start of corner to this point
            metrics['ref_trail_dist'] = ref_corner.loc[last_brake_idx, 'dist'] - start_dist
        else: metrics['ref_trail_dist'] = 0
    else: metrics['ref_trail_dist'] = 0
    
    if 'pbrake_f' in target_corner.columns:
        braking_mask = target_corner['pbrake_f'] > 1
        if braking_mask.any():
            last_brake_idx = braking_mask[::-1].idxmax()
            metrics['target_trail_dist'] = target_corner.loc[last_brake_idx, 'dist'] - start_dist
        else: metrics['target_trail_dist'] = 0

    # 6. Throttle Commitment (Distance from Apex to Full Throttle > 90%)
    # Apex is approx where speed is min
    ref_apex_dist = ref_corner.loc[ref_corner['speed'].idxmin(), 'dist']
    target_apex_dist = target_corner.loc[target_corner['speed'].idxmin(), 'dist']
    
    if 'aps' in ref_corner.columns:
        full_throttle = ref_corner[(ref_corner['dist'] > ref_apex_dist) & (ref_corner['aps'] > 90)]
        if not full_throttle.empty:
            metrics['ref_throttle_pickup'] = full_throttle.iloc[0]['dist'] - ref_apex_dist
        else: metrics['ref_throttle_pickup'] = np.nan
    else: metrics['ref_throttle_pickup'] = np.nan
    
    if 'aps' in target_corner.columns:
        full_throttle = target_corner[(target_corner['dist'] > target_apex_dist) & (target_corner['aps'] > 90)]
        if not full_throttle.empty:
            metrics['target_throttle_pickup'] = full_throttle.iloc[0]['dist'] - target_apex_dist
        else: metrics['target_throttle_pickup'] = np.nan

    # 5. Time Delta in Corner
    # Calculate time taken to traverse this segment
    ref_time = ref_corner['time_sec'].max() - ref_corner['time_sec'].min()
    target_time = target_corner['time_sec'].max() - target_corner['time_sec'].min()
    
    metrics['ref_time'] = ref_time
    metrics['target_time'] = target_time
    metrics['time_delta'] = target_time - ref_time
    
    # 6. Distance Delta (Path Length)
    # Calculate actual distance traveled (arc length)
    # We can use the 'offset' if available, or just sum the Euclidean distances between points
    ref_path_len = np.sum(np.sqrt(np.diff(ref_corner['x'])**2 + np.diff(ref_corner['y'])**2))
    target_path_len = np.sum(np.sqrt(np.diff(target_corner['x'])**2 + np.diff(target_corner['y'])**2))
    metrics['dist_delta'] = target_path_len - ref_path_len
    
    return metrics

def plot_turn_zoom(ref_df, target_df, turn_name, start_dist, end_dist, ref_driver, target_driver):
    """Plots a zoomed-in view of the racing lines AND speed trace for alignment check."""
    
    # Filter data for the turn
    ref_turn = ref_df[(ref_df['dist'] >= start_dist) & (ref_df['dist'] <= end_dist)]
    target_turn = target_df[(target_df['dist'] >= start_dist) & (target_df['dist'] <= end_dist)]
    
    if ref_turn.empty or target_turn.empty:
        return plt.figure()
        
    # Create figure with 2 subplots (Map, Speed)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=None) # Speed
    
    fig.patch.set_facecolor('black')
    
    # --- Plot 1: Racing Line Map ---
    ax1.set_facecolor('black')
    ax1.set_aspect('equal')
    
    # Plot Ref (Blue)
    ax1.plot(ref_turn['x'], ref_turn['y'], label=f'Ref ({ref_driver})', color='blue', linewidth=2.5, alpha=0.8)
    
    # Plot Target (Red)
    ax1.plot(target_turn['x'], target_turn['y'], label=f'Target ({target_driver})', color='red', linewidth=2.5, alpha=0.8)
    
    ax1.set_title(f"Racing Line: {turn_name}", color='white')
    ax1.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')
    
    # --- Plot 2: Speed Trace (Alignment Check) ---
    ax2.set_facecolor('black')
    ax2.plot(ref_turn['dist'], ref_turn['speed'], label=f'Ref', color='blue', linewidth=2)
    ax2.plot(target_turn['dist'], target_turn['speed'], label=f'Target', color='red', linewidth=2, linestyle='--')
    
    ax2.set_title("Speed Alignment Check", color='white')
    ax2.set_ylabel("Speed (km/h)", color='white')
    ax2.set_xlabel("Station (m)", color='white')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    
    ax2.set_title("Speed Alignment Check", color='white')
    ax2.set_ylabel("Speed (km/h)", color='white')
    ax2.set_xlabel("Station (m)", color='white')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig

def plot_time_gap_map(ref_df, target_df, ref_driver, target_driver, sector_data=None, sector_indices=None):
    """Plots the track map colored by Instantaneous Time Loss Rate."""
    if ref_df.empty or target_df.empty: return plt.figure()
    
    # Merge on distance
    if 'dist' not in ref_df.columns or 'dist' not in target_df.columns: return plt.figure()
    
    merged = pd.merge_asof(target_df.sort_values('dist'), ref_df.sort_values('dist'), on='dist', suffixes=('_target', '_ref'), direction='nearest')
    
    # Calculate Time Delta (Target Time - Ref Time)
    # We now have 'time_sec' from process_telemetry_data which is relative seconds from start
    
    if 'time_sec' not in merged.columns and 'time_sec_ref' in merged.columns:
         # merged suffixes
         pass
         
    # merged has time_sec_target and time_sec_ref
    merged['time_delta'] = merged['time_sec_target'] - merged['time_sec_ref']
    
    # Calculate Time Loss Rate analytically using Speed
    # d(Time)/d(Dist) = 1/v
    # Rate = d(Gap)/d(Dist) = (1/v_target - 1/v_ref)
    # This shows EXACTLY where time is gained/lost (e.g. in a corner)
    
    # Ensure speed is in m/s (assuming input is km/h)
    v_ref = merged['speed_ref'] / 3.6
    v_target = merged['speed_target'] / 3.6
    
    # Avoid division by zero (clip low speeds to 10 km/h approx 3 m/s)
    v_ref = np.maximum(v_ref, 3.0)
    v_target = np.maximum(v_target, 3.0)
    
    # Calculate Loss Rate (s/m)
    loss_rate_per_m = (1.0 / v_target) - (1.0 / v_ref)
    
    # Scale to s/100m for readability
    z = loss_rate_per_m * 100
    
    # Smooth the data to reduce noise and make zones clearer
    from scipy.signal import savgol_filter
    try:
        if len(z) > 21:
            z = savgol_filter(z, window_length=21, polyorder=3)
    except: pass
    
    # Dynamic Normalization (Robust)
    # Use percentiles to avoid outliers skewing the scale
    # We want a symmetric scale centered at 0
    vmin, vmax = np.percentile(z, [5, 95])
    limit = max(abs(vmin), abs(vmax))
    
    # Clamp the limit to reasonable bounds to prevent:
    # 1. Washed out map (limit too high) -> Cap at 0.5
    # 2. Saturated map (limit too low) -> Floor at 0.05
    limit = np.clip(limit, 0.05, 0.5)
    
    norm = plt.Normalize(-limit, limit)

    x = merged['x_target'].values
    y = merged['y_target'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Custom Colormap: Green (Gaining) -> White (Neutral) -> Red (Losing)
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.0, '#00B050'), (0.5, '#ffffff'), (1.0, '#ff4d4d')] # Green -> White -> Red
    cmap_name = 'GreenWhiteRed'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Normalize: Dynamic
    # norm = plt.Normalize(-0.1, 0.1)
    
    lc = LineCollection(segments, cmap=cm, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)
    
    ax.set_xlim(x.min() - 50, x.max() + 50)
    ax.set_ylim(y.min() - 50, y.max() + 50)
    
    # Annotations
    if sector_data:
        for label, (lx, ly, delta) in sector_data.items():
            color = '#ff4d4d' if delta > 0 else '#00B050' # Red if positive (loss), Green if negative (gain)
            txt = f"{label}\n{delta:+.2f}s"
            ax.text(lx, ly, txt, color=color, fontsize=11, ha='center', va='center', fontweight='bold', zorder=10,
                    bbox=dict(facecolor='black', edgecolor=color, alpha=0.7, pad=2))

    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label(f"Time Loss Rate (s/100m)\nRed = Losing Time, Green = Gaining Time", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(f"Instantaneous Time Loss: {target_driver} vs {ref_driver}", color='white', fontsize=14)
    return fig

def plot_telemetry_comparison(ref_df, target_df, ref_driver, target_driver):
    """Plots Speed, Throttle, and Brake comparisons."""
    if ref_df.empty or target_df.empty: return plt.figure()
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.patch.set_facecolor('black')
    
    # Speed
    ax = axes[0]
    ax.set_facecolor('black')
    ax.plot(ref_df['dist'], ref_df['speed'], label=f'Ref ({ref_driver})', color='blue', linewidth=2)
    ax.plot(target_df['dist'], target_df['speed'], label=f'Target ({target_driver})', color='red', linewidth=2, linestyle='--')
    ax.set_ylabel("Speed (km/h)", color='white')
    ax.set_title("Speed Trace", color='white')
    ax.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    # Throttle
    ax = axes[1]
    ax.set_facecolor('black')
    if 'aps' in ref_df.columns and 'aps' in target_df.columns:
        ax.plot(ref_df['dist'], ref_df['aps'], label=f'Ref ({ref_driver})', color='blue', linewidth=2)
        ax.plot(target_df['dist'], target_df['aps'], label=f'Target ({target_driver})', color='red', linewidth=2, linestyle='--')
    ax.set_ylabel("Throttle (%)", color='white')
    ax.set_title("Throttle Trace", color='white')
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    # Brake
    ax = axes[2]
    ax.set_facecolor('black')
    if 'pbrake_f' in ref_df.columns and 'pbrake_f' in target_df.columns:
        ax.plot(ref_df['dist'], ref_df['pbrake_f'], label=f'Ref ({ref_driver})', color='blue', linewidth=2)
        ax.plot(target_df['dist'], target_df['pbrake_f'], label=f'Target ({target_driver})', color='red', linewidth=2, linestyle='--')
    ax.set_ylabel("Brake (bar)", color='white')
    ax.set_title("Brake Trace", color='white')
    ax.set_xlabel("Distance (m)", color='white')
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig


def plot_coasting(df, driver_name):
    """Highlights coasting zones."""
    if df.empty: return plt.figure()
    x = df['x'].values
    y = df['y'].values
    is_coasting = (df['aps'] < 5) & (df['pbrake_f'] < 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.plot(x, y, color='gray', linewidth=2, alpha=0.5)
    coasting_x = x[is_coasting]
    coasting_y = y[is_coasting]
    
    if len(coasting_x) > 0:
        ax.scatter(coasting_x, coasting_y, color='yellow', s=10, label='Coasting', zorder=5)
        
    ax.set_title(f"Coasting Zones: {driver_name}", color='white')
    ax.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    return fig

def plot_delta_bar(delta_df, target_driver, ref_driver):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    delta_values = delta_df['Delta']
    colors = ['#00B050' if d <= 0 else '#ff4d4d' for d in delta_values] 
    ax.bar(delta_df['Sector'], delta_values, color=colors)
    
    ax.set_ylabel('Delta Time (s)', color='white')
    ax.set_title(f'Sector Time Difference: {target_driver} vs. {ref_driver}', color='white')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.tight_layout()
    return fig

def get_color(delta, max_loss):
    if pd.isna(delta) or delta == 0: return 'gray'
    if delta < 0: return '#00B050' 
    else:
        max_loss = max(0.001, max_loss) 
        r_val = min(1.0, 0.4 + 0.6 * (delta / max_loss)) 
        return (r_val, 0.2, 0.2) 

def plot_map(df, centerline_points, sector_indices, track_name, target_driver, ref_driver, sector_annots, sub_annots):
    if df.empty or not centerline_points: return plt.figure()
    
    MAX_LOSS = df['Delta'].loc[df['Delta'] > 0].max() if (df['Delta'] > 0).any() else 0.5
    xs = [p[0] for p in centerline_points]
    ys = [p[1] for p in centerline_points]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    buffer = 50
    ax.set_xlim(min(xs) - buffer, max(xs) + buffer)
    ax.set_ylim(min(ys) - buffer, max(ys) + buffer)
    if track_name == 'COTA':
         ax.set_ylim(max(ys) + buffer, min(ys) - buffer)

    ax.plot(xs, ys, color='gray', linewidth=3, linestyle='--', alpha=0.5, zorder=1)
    fig.suptitle(f"{track_name}: Driver {target_driver} vs. {ref_driver}", color='white', fontsize=14, y=0.95)
    
    num_segments = min(len(sector_indices), len(df))
    for i in range(num_segments):
        start, end = sector_indices[i]
        segment_points = centerline_points[start:end+1]
        row = df.iloc[i]
        color = get_color(row['Delta'], MAX_LOSS)
        sx = [p[0] for p in segment_points]
        sy = [p[1] for p in segment_points]
        ax.plot(sx, sy, color=color, linewidth=15, solid_capstyle='round', alpha=0.7, zorder=2)
        
        mid_idx = len(segment_points) // 2
        mid_x, mid_y = segment_points[mid_idx]
        delta_text = f"{row['Delta']:+.2f}s" if not pd.isna(row['Delta']) else "N/A"
        ax.text(mid_x + 20, mid_y + 20, delta_text, color='white', fontsize=9, ha='center', va='center', fontweight='bold', zorder=3)

    if sector_annots:
        for _, ((x, y), label) in sector_annots.items():
            ax.text(x, y, label, color='yellow', fontsize=18, ha='center', va='center', fontweight='bold', zorder=5)
    
    if centerline_points:
        sx, sy = centerline_points[0]
        ax.plot([sx], [sy], marker='o', color='white', markersize=10, zorder=4)
        ax.text(sx - 30, sy + 10, 'S/F', color='yellow', fontsize=10, ha='center', va='center', fontweight='bold', zorder=4)

    return fig

def calculate_delta(target_row, ref_row, micro_sectors):
    delta_results = []
    accumulated_delta = 0.0
    if target_row is None or ref_row is None: return None, None
        
    for sector_name, time_col in micro_sectors:
        sec_col = time_col + '_SEC'
        if sec_col not in target_row or sec_col not in ref_row: continue
        time_ref = ref_row[sec_col]
        time_target = target_row[sec_col]
        
        if pd.isna(time_target) or pd.isna(time_ref): sector_delta = np.nan
        else: sector_delta = time_target - time_ref
        
        if not pd.isna(sector_delta): accumulated_delta += sector_delta
        
        delta_results.append({
            'Sector': sector_name,
            'Time_Target': time_target,
            'Time_Reference': time_ref,
            'Delta': sector_delta,
            'Accumulated_Delta': accumulated_delta
        })

    delta_df = pd.DataFrame(delta_results)
    if pd.isna(target_row['LAP_TIME_SEC']) or pd.isna(ref_row['LAP_TIME_SEC']): total_delta = np.nan
    else: total_delta = target_row['LAP_TIME_SEC'] - ref_row['LAP_TIME_SEC']
    return delta_df, total_delta

# ------------------------
# Main App
# ------------------------

# ------------------------
# Main App
# ------------------------

def main():
    st.title("Toyota Racing Analysis Dashboard")
    
    # Sidebar: Track Selection
    track_name = st.sidebar.selectbox("Select Track", list(TRACK_CONFIG.keys()))
    config = TRACK_CONFIG[track_name]
    
    # Load Endurance Data
    endurance_df = load_race_data(config['endurance_file'], config['micro_sectors'])
    if endurance_df.empty: st.stop()

    # Basic filter: drop obvious in/out laps to keep UI clean
    apply_duration_filter = st.sidebar.checkbox("Exclude laps <60s or >200s", value=True)
    if apply_duration_filter:
        endurance_df = filter_laps_by_duration(endurance_df, min_sec=60, max_sec=200)
        if endurance_df.empty:
            st.warning("All laps were filtered out; relax the duration filter.")
            st.stop()
    
    driver_numbers = sorted(endurance_df['NUMBER'].unique().tolist())
    
    # Sidebar: Analysis Mode (From multi_track_analysis.py)
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Driver vs. Fastest Lap", "Head-to-Head"])
    
    # Determine Target and Reference based on Mode
    if analysis_mode == "Driver vs. Fastest Lap":
        target_driver = st.sidebar.selectbox("Select Driver", driver_numbers)
        
        # Find overall fastest
        fastest_idx = endurance_df['LAP_TIME_SEC'].idxmin()
        ref_driver = endurance_df.loc[fastest_idx, 'NUMBER']
        
        # Get Data Rows
        target_laps = endurance_df[endurance_df['NUMBER'] == target_driver].sort_values('LAP_TIME_SEC')
        if target_laps.empty:
            st.error("No laps remain for this driver after filtering.")
            st.stop()
        target_row = target_laps.iloc[0]
        ref_row = endurance_df.loc[fastest_idx]
        
        # For Insights Tab (Fastest Laps)
        target_lap_num = target_row['LAP_NUMBER']
        ref_lap_num = ref_row['LAP_NUMBER']
        
    elif analysis_mode == "Head-to-Head":
        col_a, col_b = st.sidebar.columns(2)
        driver_a = col_a.selectbox("Driver A", driver_numbers, index=0)
        driver_b = col_b.selectbox("Driver B", driver_numbers, index=min(1, len(driver_numbers)-1))
        
        target_driver = driver_a
        ref_driver = driver_b
        
        # Get Data Rows (Fastest lap for each driver)
        target_laps = endurance_df[endurance_df['NUMBER'] == target_driver].sort_values('LAP_TIME_SEC')
        ref_laps = endurance_df[endurance_df['NUMBER'] == ref_driver].sort_values('LAP_TIME_SEC')
        
        if target_laps.empty or ref_laps.empty:
            st.error("No data for selected drivers")
            st.stop()
            
        target_row = target_laps.iloc[0]
        ref_row = ref_laps.iloc[0]
        
        target_lap_num = target_row['LAP_NUMBER']
        ref_lap_num = ref_row['LAP_NUMBER']

    # Top-row KPIs (adapts to mode)
    render_top_kpis(analysis_mode, target_row, ref_row)

    # Tabs
    tab1, tab_playback, tab_profile, tab2 = st.tabs(["Sector Analysis", "Playback", "Driver Profile", "Advanced Insights (Telemetry)"])
    
    # --- Tab 1: Sector Analysis (Exact match to multi_track_analysis.py) ---
    with tab1:
        st.header(f"Sector Analysis: {track_name}")
        
        # Load Track Map
        if config['type'] == 'pixel':
            centerline_points, sector_indices = load_cota_track()
        else:
            centerline_points, sector_indices = load_barber_track(config['telemetry_ref_file'])
            
        # Calculate Deltas
        delta_df, total_delta = calculate_delta(target_row, ref_row, config['micro_sectors'])
        st.session_state['official_delta_df'] = delta_df # Store for Tab 2
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        if analysis_mode == "Driver vs. Fastest Lap":
            # Get Rank
            target_rank = target_row['RANK']
            def get_rank_label(r):
                if pd.isna(r): return "N/A"
                ri = int(r)
                sf = 'th'
                if ri % 10 == 1 and ri % 100 != 11: sf = 'st'
                elif ri % 10 == 2 and ri % 100 != 12: sf = 'nd'
                elif ri % 10 == 3 and ri % 100 != 13: sf = 'rd'
                return f"{ri}{sf} Fastest"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Delta vs Fastest", f"{total_delta:+.3f} s", get_rank_label(target_rank))
                if delta_df is not None:
                    st.dataframe(delta_df[['Sector', 'Delta', 'Accumulated_Delta']].style.format({'Delta': '{:+.3f}'.format, 'Accumulated_Delta': '{:+.3f}'.format}))
                    st.pyplot(plot_delta_bar(delta_df, target_driver, ref_driver))
            with col2:
                if delta_df is not None:
                    st.pyplot(plot_map(delta_df, centerline_points, sector_indices, track_name, target_driver, ref_driver, config['sector_annotations'], config['subsection_annotations']))
                    
        elif analysis_mode == "Head-to-Head":
            # Get Ranks
            rank_a = target_row['RANK']
            rank_b = ref_row['RANK']
            def get_rank_label_simple(r): return f"{int(r)}" if not pd.isna(r) else "N/A"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Delta (A vs B)", f"{total_delta:+.3f} s")
                st.write(f"Driver A Rank: {get_rank_label_simple(rank_a)}")
                st.write(f"Driver B Rank: {get_rank_label_simple(rank_b)}")
                if delta_df is not None:
                    st.dataframe(delta_df[['Sector', 'Delta', 'Accumulated_Delta']].style.format({'Delta': '{:+.3f}'.format, 'Accumulated_Delta': '{:+.3f}'.format}))
                    st.pyplot(plot_delta_bar(delta_df, target_driver, ref_driver))
            with col2:
                if delta_df is not None:
                    st.pyplot(plot_map(delta_df, centerline_points, sector_indices, track_name, target_driver, ref_driver, config['sector_annotations'], config['subsection_annotations']))

        # Fastest-lap Simulation (telemetry-driven playback)
        st.divider()
        st.subheader("Fastest-lap Simulation")
        fastest_sim, tele_sim, track_poly = load_fastest_assets(track_name)
        if fastest_sim.empty or tele_sim.empty or track_poly is None:
            st.info("Simulation data not available for this track.")
        else:
            cars = list(dict.fromkeys([target_driver, ref_driver]))
            cars = [c for c in cars if not pd.isna(c)]
            s1_frac, s2_frac = 0.33, 0.66
            if set(["S1_SEC", "S2_SEC", "S3_SEC"]).issubset(fastest_sim.columns):
                candidate = fastest_sim[
                    fastest_sim["CAR_NUMBER"].isin(cars)
                    & fastest_sim[["S1_SEC", "S2_SEC", "S3_SEC"]].applymap(np.isfinite).all(axis=1)
                ].head(1)
                if not candidate.empty:
                    total = candidate[["S1_SEC", "S2_SEC", "S3_SEC"]].iloc[0].sum()
                    if np.isfinite(total) and total > 0:
                        s1_frac = float(candidate["S1_SEC"].iloc[0] / total)
                        s2_frac = float((candidate["S1_SEC"].iloc[0] + candidate["S2_SEC"].iloc[0]) / total)
            sim_df = build_sim_frames(fastest_sim, tele_sim, cars, track_poly)
            available = set(sim_df["vehicle_number"].astype(str)) if not sim_df.empty else set()
            missing = [c for c in cars if str(c) not in available]
            if missing:
                st.warning(f"No simulation telemetry found for: {', '.join(map(str, missing))}. Showing available cars only.")
            if sim_df.empty:
                st.info("No telemetry slices found for selected drivers' fastest laps.")
            else:
                fig = simulation_plot(sim_df, track_poly, (s1_frac, s2_frac))
                st.plotly_chart(fig, width="stretch")

    # --- Tab Playback: Moving dots + synced plots ---
    with tab_playback:
        render_playback_section(analysis_mode, track_name, config, ref_driver, target_driver, ref_row, target_row, ref_lap_num, target_lap_num)

    # --- Tab Driver Profile: Radar + stats ---
    with tab_profile:
        render_driver_profile_section(track_name, config)

    # --- Tab 2: Advanced Insights (Exact match to barber_insights.py) ---
    with tab2:
        if config['telemetry_file']:
            st.header(f"Advanced Insights: {track_name}")
            st.markdown("Actionable insights: Heatmaps, Racing Lines, and Coasting Detection.")
            
            # Show selected laps
            st.info(f"Analyzing: Driver {target_driver} (Lap {target_lap_num}) vs Reference {ref_driver} (Lap {ref_lap_num})")
            
            # Load Button (State Managed)
            if st.button("Load & Analyze Telemetry", type="primary"):
                with st.spinner(f"Aligning telemetry for Driver {ref_driver} vs {target_driver}..."):
                    
                    # 1. Find matching laps in telemetry (Smart Matching)
                    ref_target_time = ref_row['LAP_TIME_SEC']
                    target_target_time = target_row['LAP_TIME_SEC']
                    
                    real_ref_lap, ref_diff = find_closest_telemetry_lap(config['telemetry_file'], ref_driver, ref_target_time, target_lap_num=ref_lap_num)
                    real_target_lap, target_diff = find_closest_telemetry_lap(config['telemetry_file'], target_driver, target_target_time, target_lap_num=target_lap_num)
                    
                    if real_ref_lap is None:
                        st.warning(f"Could not find telemetry for Ref Driver {ref_driver}. Using Endurance Lap {ref_lap_num}.")
                        real_ref_lap = ref_lap_num
                    else:
                         st.caption(f"Using Telemetry Lap {real_ref_lap} for Ref Driver (Endurance Lap {ref_lap_num}: {ref_target_time:.2f}s vs Telemetry: {ref_target_time + (ref_diff if ref_diff else 0):.2f}s)")
                    
                    if real_target_lap is None:
                        st.warning(f"Could not find telemetry for Target Driver {target_driver}. Using Endurance Lap {target_lap_num}.")
                        real_target_lap = target_lap_num
                    else:
                         st.caption(f"Using Telemetry Lap {real_target_lap} for Target Driver (Endurance Lap {target_lap_num}: {target_target_time:.2f}s vs Telemetry: {target_target_time + (target_diff if target_diff else 0):.2f}s)")

                    # 2. Load Data
                    ref_raw = load_telemetry_for_lap(config['telemetry_file'], ref_driver, real_ref_lap)
                    target_raw = load_telemetry_for_lap(config['telemetry_file'], target_driver, real_target_lap)
                    
                    if ref_raw.empty or target_raw.empty:
                        st.error("Telemetry data not found for one or both drivers.")
                        st.session_state['data_loaded'] = False
                    else:
                        # --- NEW: Track Projection Alignment ---
                        st.session_state['ref_telemetry'] = process_telemetry_data(ref_raw)
                        target_processed = process_telemetry_data(target_raw)
                        
                        # 1. Generate Master Centerline from Reference Lap
                        # We use the reference lap as the "Centerline" definition
                        centerline_data = track_projection.generate_centerline(st.session_state['ref_telemetry'])
                        
                        # 2. Project Reference to Centerline (Self-projection to get Station)
                        ref_proj = track_projection.project_to_centerline(st.session_state['ref_telemetry'], centerline_data)
                        
                        # 3. Project Target to Centerline
                        target_proj = track_projection.project_to_centerline(target_processed, centerline_data)
                        
                        # 4. Resample both to common Station Grid
                        # We use the max station of the reference lap as the limit
                        max_station = ref_proj['station'].max()
                        station_grid = np.arange(0, max_station, 1.0) # 1m grid
                        
                        ref_aligned = track_projection.align_data_to_station(ref_proj, station_grid)
                        target_aligned = track_projection.align_data_to_station(target_proj, station_grid)
                        
                        # Update Session State
                        # We rename 'station' to 'dist' so existing functions work without change
                        ref_aligned['dist'] = ref_aligned['station']
                        target_aligned['dist'] = target_aligned['station']
                        
                        st.session_state['ref_telemetry'] = ref_aligned
                        st.session_state['target_telemetry'] = target_aligned
                        
                        st.session_state['telemetry_loaded'] = True
                        st.session_state['loaded_drivers'] = (ref_driver, target_driver)
            
            if st.session_state.get('telemetry_loaded', False):
                ref_data = st.session_state['ref_telemetry']
                target_data = st.session_state['target_telemetry']
                # Ensure we are viewing the drivers we loaded
                loaded_ref, loaded_target = st.session_state.get('loaded_drivers', (None, None))
                
                # 1. Time Gap Analysis (New)
                # 1. Time Gap Analysis (New)
                # Header removed as per user request
                
                # Calculate Sector Deltas for Annotation
                # CRITICAL: Use delta_df from Tab 1 (Endurance Data) to ensure consistency with the table.
                # We only use Telemetry for the heatmap colors (visual trend), but the numbers must match the official analysis.
                sector_deltas = {}
                if track_name == 'Barber' and delta_df is not None:
                    cl_points, sec_indices = load_barber_track(config['telemetry_ref_file'])
                    sector_names = [s[0] for s in config['micro_sectors']]
                    
                    # Map Sector Name -> Accumulated Delta from delta_df
                    # delta_df has columns: Sector, Delta, Accumulated_Delta
                    # We want Accumulated_Delta at the end of the sector
                    
                    total_points = len(cl_points)
                    
                    if len(sec_indices) == len(sector_names):
                        for i, (start, end) in enumerate(sec_indices):
                            sec_name = sector_names[i]
                            
                            # Find matching row in delta_df
                            row = delta_df[delta_df['Sector'] == sec_name]
                            if not row.empty:
                                delta_val = row.iloc[0]['Accumulated_Delta']
                                
                                # Get coords for label (midpoint of sector)
                                mid_idx = (start + end) // 2
                                if mid_idx < len(cl_points):
                                    mx, my = cl_points[mid_idx]
                                    sector_deltas[sec_name] = (mx, my, delta_val)

                st.header("1. Racing Line & Speed Comparison")
                st.info("Visual comparison of driving lines. Blue = Reference, Red = Target. Look for Wide vs Tight lines.")
                
                st.pyplot(plot_dual_speed_map(st.session_state['ref_telemetry'], st.session_state['target_telemetry'], ref_driver, target_driver))
                


                # 2. Heatmaps
                st.header("3. Telemetry Heatmaps")
                st.info("Select a metric to visualize its intensity around the track. Red/Yellow = High, Blue/Purple = Low.")
                
                heatmap_metric = st.selectbox(
                    "Select Heatmap Metric", 
                    ['pbrake_f', 'aps'], 
                    format_func=lambda x: {'pbrake_f': 'Brake Pressure (bar)', 'aps': 'Throttle (%)'}[x]
                )
                
                # dynamic_scaling = st.checkbox("Enable Dynamic Scaling (Maximize Contrast)", value=False)
                dynamic_scaling = False # User requested to remove the contrast option, defaulting to False or handling internally
                show_sectors = st.checkbox("Show Sector Labels", value=True)
                
                # Prepare Annotations
                heatmap_annotations = []
                if show_sectors:
                    if track_name == 'Barber':
                        # Generate dynamic annotations for Barber
                        # We need centerline_points and sector_indices from the load function
                        # But those are cached, so we can call them again cheaply
                        cl_points, sec_indices = load_barber_track(config['telemetry_ref_file'])
                        sector_names = [s[0] for s in config['micro_sectors']] # e.g. S1-a, S1-b...
                        
                        if len(sec_indices) == len(sector_names):
                            for i, (start, end) in enumerate(sec_indices):
                                mid_idx = (start + end) // 2
                                if mid_idx < len(cl_points):
                                    mx, my = cl_points[mid_idx]
                                    heatmap_annotations.append((mx, my, sector_names[i]))
                    elif track_name == 'COTA':
                        # Use hardcoded subsections
                        for label, ((ix, iy), _) in config['subsection_annotations'].items():
                            heatmap_annotations.append((ix, iy, label))

                c1, c2 = st.columns(2)
                with c1: st.pyplot(plot_heatmap(ref_data, heatmap_metric, f"Ref: {loaded_ref} ({heatmap_metric})", cmap='plasma', dynamic_scaling=dynamic_scaling, annotations=heatmap_annotations))
                with c2: st.pyplot(plot_heatmap(target_data, heatmap_metric, f"Target: {loaded_target} ({heatmap_metric})", cmap='plasma', dynamic_scaling=dynamic_scaling, annotations=heatmap_annotations))
                

                
                # 2. Turn Zoom
                st.header("2. Racing Line Comparison (Turn Zoom)")
                st.info("Zoom into specific corners to compare driving lines.")
                
                if track_name == 'Barber':
                    # Auto-detect turns if not already done
                    if 'detected_turns' not in st.session_state:
                        turns = detect_turns(st.session_state['ref_telemetry'])
                        
                        # Specific Merge for Turn 5 and 6 (User Request)
                        if "Turn 5" in turns and "Turn 6" in turns:
                            t5_start, t5_end = turns["Turn 5"]
                            t6_start, t6_end = turns["Turn 6"]
                            
                            # Create merged turn
                            turns["Turn 5-6"] = (t5_start, t6_end)
                            
                            # Remove individual turns
                            del turns["Turn 5"]
                            del turns["Turn 6"]
                            
                            # Sort turns by start distance
                            sorted_turns = dict(sorted(turns.items(), key=lambda item: item[1][0]))
                            st.session_state['detected_turns'] = sorted_turns
                        else:
                            st.session_state['detected_turns'] = turns
                    
                    detected_turns = st.session_state['detected_turns']
                    
                    if detected_turns:
                        turn_name = st.selectbox("Select Turn to Zoom", list(detected_turns.keys()))
                        start_dist, end_dist = detected_turns[turn_name]
                    else:
                        st.warning("Could not auto-detect turns. Using default full lap.")
                        turn_name = "Full Lap"
                        start_dist, end_dist = 0, st.session_state['ref_telemetry']['dist'].max()
                    
                    # Run Advanced Analysis
                    # metrics = analyze_corner_performance(ref_data, target_data, start_dist, end_dist)
                    
                    # if metrics:
                    #     st.markdown(f"### Deep Dive: {turn_name}")
                    #     ... (Removed Metrics Table) ...
                    
                    st.pyplot(plot_turn_zoom(ref_data, target_data, turn_name, start_dist, end_dist, loaded_ref, loaded_target))
                else:
                    st.warning("Turn definitions not available for this track.")
                
                # 3. Coasting
                st.header("3. Coasting Detection (Hesitation)")
                st.info("Yellow dots indicate areas where the driver is **coasting** (Throttle < 5% AND Brake < 1 bar).")
                
                c3, c4 = st.columns(2)
                with c3: st.pyplot(plot_coasting(ref_data, loaded_ref))
                with c4: st.pyplot(plot_coasting(target_data, loaded_target))
                
        else:
            st.info("Advanced telemetry analysis is not available for this track yet.")

    # Data status at bottom
    st.divider()
    render_data_status(config)

if __name__ == "__main__":
    main()
