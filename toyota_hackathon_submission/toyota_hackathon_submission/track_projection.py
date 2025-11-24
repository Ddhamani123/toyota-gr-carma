import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree

def generate_centerline(ref_df, smoothing=0.5):
    """
    Generates a high-resolution centerline spline from the reference lap.
    
    Args:
        ref_df (pd.DataFrame): Reference lap data with 'x' and 'y' columns.
        smoothing (float): Smoothing factor for the spline.
        
    Returns:
        dict: 'tck' (spline parameters), 'u' (parameter values), 'points' (N, 2 array of points), 'length' (float)
    """
    # Extract coordinates
    x = ref_df['x'].values
    y = ref_df['y'].values
    
    # Remove duplicates to avoid spline errors
    points = np.vstack((x, y)).T
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(unique_indices)]
    
    # Fit spline
    # s=0 means no smoothing (pass through all points), s>0 smooths
    # per=1 means periodic (closed loop)
    tck, u = splprep(points.T, s=smoothing, per=1)
    
    # Generate high-res points for KDTree (e.g. every 0.1m approx)
    # We don't know exact length yet, but let's oversample
    u_fine = np.linspace(0, 1, len(points) * 10)
    x_fine, y_fine = splev(u_fine, tck)
    fine_points = np.vstack((x_fine, y_fine)).T
    
    # Calculate total length
    diffs = np.diff(fine_points, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    total_length = dists.sum()
    
    return {
        'tck': tck,
        'fine_points': fine_points,
        'length': total_length,
        'kdtree': KDTree(fine_points)
    }

def project_to_centerline(telemetry_df, centerline_data):
    """
    Projects telemetry points onto the centerline to find Station (dist along line) and Offset.
    
    Args:
        telemetry_df (pd.DataFrame): Data with 'x' and 'y'.
        centerline_data (dict): Output from generate_centerline.
        
    Returns:
        pd.DataFrame: Original df with 'station' and 'offset' columns added.
    """
    points = np.vstack((telemetry_df['x'].values, telemetry_df['y'].values)).T
    tree = centerline_data['kdtree']
    fine_points = centerline_data['fine_points']
    total_length = centerline_data['length']
    
    # Find nearest point on fine-grained centerline
    dists, indices = tree.query(points)
    
    # Calculate Station (Distance along centerline)
    # We approximate station by the cumulative distance of the fine points
    # Pre-calculate cumulative distance for fine points
    fine_diffs = np.diff(fine_points, axis=0, prepend=fine_points[[0]])
    fine_dists = np.sqrt((fine_diffs**2).sum(axis=1))
    fine_cumdist = np.cumsum(fine_dists)
    
    stations = fine_cumdist[indices]
    
    # Calculate Offset (Signed distance)
    # Cross product of tangent vector and vector to point
    # Tangent at matched point
    # We can use the spline derivative, but finite difference of fine points is faster/easier
    
    # Vector from centerline point to car
    # matched_points = fine_points[indices]
    # vec_to_car = points - matched_points
    
    # Tangent vector at matched point (approximate)
    # next_indices = (indices + 1) % len(fine_points)
    # tangents = fine_points[next_indices] - fine_points[indices]
    
    # Normalize tangents
    # norms = np.linalg.norm(tangents, axis=1)
    # tangents = tangents / norms[:, np.newaxis]
    
    # Cross product (2D): x1*y2 - x2*y1
    # cross = tangents[:, 0] * vec_to_car[:, 1] - tangents[:, 1] * vec_to_car[:, 0]
    
    # Offset is distance * sign(cross)
    # offset = dists * np.sign(cross)
    
    # Simplified: Just return station for now, offset calculation can be added if needed for "wide/tight" analysis
    # But for alignment, Station is key.
    
    df_out = telemetry_df.copy()
    df_out['station'] = stations
    df_out['offset'] = dists # Absolute distance for now
    
    # Sort by station to ensure monotonicity (handle noise/loops)
    df_out = df_out.sort_values('station')
    
    return df_out

def align_data_to_station(df, station_grid):
    """
    Resamples data to a common Station grid.
    """
    # Remove duplicates in station
    df = df.drop_duplicates(subset=['station'])
    
    resampled = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            resampled[col] = np.interp(station_grid, df['station'], df[col])
            
    res_df = pd.DataFrame(resampled)
    res_df['station'] = station_grid
    return res_df
