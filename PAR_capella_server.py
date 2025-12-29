# PAR_capella_server.py - FIXED: Tomogram displays correctly with topography
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import plotly
import plotly.graph_objs as go
import numpy as np
import logging
import traceback
import os
from datetime import datetime
from PAR_capella_filters import apply_processing_pipeline
from PAR_capella_topography import (
    warp_tomogram_to_topography,
    create_topography_mesh,
    add_vertical_exaggeration,
    resample_profile_to_width,
    validate_topography_data,
    create_flat_topography,
    create_topography_mesh_correct,
    prepare_topographic_display_simple
)

app = Flask(__name__, template_folder='.')
GLOBAL_DATA = None

def start_flask_server(data_container, port=5000):
    global GLOBAL_DATA
    GLOBAL_DATA = data_container
    # Lower log level to let our prints show up clearly
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(f"   ✓ Server listening on port {port}...")
    app.run(port=port, debug=False)

@app.route('/')
def index():
    if GLOBAL_DATA is None: 
        return "<h3>No Data Loaded.</h3>"
    return render_template('PAR-index.html')

def get_surface_relative_vector(data_shape):
    """
    Get surface-relative vector for tomogram display.
    CORRECTED: z_vec is depth/height relative to surface (0 = surface).
    Negative values = below surface (depth), Positive values = above surface (height).
    
    Returns values in meters relative to surface (0 = ground level).
    """
    # Try to get z_vec from data
    if 'z_vec' in GLOBAL_DATA and GLOBAL_DATA['z_vec'] is not None:
        z_vec_raw = GLOBAL_DATA['z_vec'].copy()
        
        print(f"\n   🔍 DEBUG: SURFACE-RELATIVE VECTOR ANALYSIS")
        print(f"      Shape: {z_vec_raw.shape}")
        print(f"      Type: {z_vec_raw.dtype}")
        print(f"      First 5 values: {z_vec_raw[:5]}")
        print(f"      Last 5 values: {z_vec_raw[-5:]}")
        print(f"      Min: {np.min(z_vec_raw):.1f}m, Max: {np.max(z_vec_raw):.1f}m")
        print(f"      Mean: {np.mean(z_vec_raw):.1f}m")
        
        # Check orientation - should go from negative (deep) to positive (shallow/above)
        if z_vec_raw[0] < z_vec_raw[-1]:
            print(f"   ✅ Vector increases upward (negative → positive)")
            print(f"      Deepest point: {z_vec_raw[0]:.1f}m (below surface)")
            print(f"      Highest point: {z_vec_raw[-1]:.1f}m (above surface)")
            
            # This is the CORRECT orientation
            # -500m = 500m below surface
            # +10m = 10m above surface
            # 0m = ground surface
            
            # Calculate depth/height ranges
            below_surface = z_vec_raw[z_vec_raw < 0]
            above_surface = z_vec_raw[z_vec_raw > 0]
            
            if len(below_surface) > 0:
                max_depth = abs(np.min(below_surface))
                print(f"   ⬇️  Maximum depth below surface: {max_depth:.1f}m")
            
            if len(above_surface) > 0:
                max_height = np.max(above_surface)
                print(f"   ⬆️  Maximum height above surface: {max_height:.1f}m")
            
            print(f"   📊 Surface position: 0.0m (ground level)")
            print(f"   📈 Vector shows: {z_vec_raw[0]:.1f}m to {z_vec_raw[-1]:.1f}m relative to surface")
            
            return z_vec_raw
        else:
            print(f"   ⚠️  Vector decreases upward - reversing")
            z_vec_reversed = z_vec_raw[::-1]
            print(f"   🔄 Reversed: {z_vec_reversed[0]:.1f}m to {z_vec_reversed[-1]:.1f}m")
            return z_vec_reversed
    
    # Fallback: estimate surface-relative vector
    print(f"   📊 No z_vec found, estimating...")
    
    # PAR typical: 2m per bin, 256 bins = 512m total
    # Typical: -500m to +12m (512m range)
    estimated_total_range = 512.0  # meters
    start_depth = -500.0  # meters below surface
    end_height = start_depth + estimated_total_range  # -500 + 512 = +12m
    
    z_vec = np.linspace(start_depth, end_height, data_shape)
    
    print(f"   📐 Estimated surface-relative vector:")
    print(f"      {start_depth:.1f}m to {end_height:.1f}m ({data_shape} bins)")
    print(f"      Resolution: {estimated_total_range/(data_shape-1):.2f}m/bin")
    print(f"      0m = ground surface")
    
    return z_vec

def calculate_vertical_range(surface_relative_vec, use_topo, topo_profile=None, 
                           depth_min=None, depth_max=None, height_min=None, height_max=None):
    """
    Calculate the vertical display range based on user preferences and topography.
    
    Parameters:
    -----------
    surface_relative_vec : numpy array
        Vector in meters relative to surface (0 = ground)
    use_topo : bool
        Whether to use topography
    topo_profile : numpy array, optional
        Topography elevation profile
    depth_min, depth_max, height_min, height_max : float, optional
        User-defined crop limits
    
    Returns:
    --------
    dict with display range information
    """
    vec_min = np.min(surface_relative_vec)  # Most negative (deepest)
    vec_max = np.max(surface_relative_vec)  # Most positive (highest)
    total_range = vec_max - vec_min
    
    # Default to full range
    display_min = vec_min
    display_max = vec_max
    
    # Track if we need to crop the data
    crop_data = False
    crop_indices = None
    
    # Apply user crop if provided
    if depth_min is not None:
        # depth_min should be positive (e.g., 100 means show from -100m)
        crop_min = -abs(depth_min)  # Convert to negative
        if crop_min > display_min:  # crop_min is less negative
            display_min = crop_min
            print(f"   🔧 Crop: Minimum depth limited to {abs(crop_min):.0f}m below surface")
            crop_data = True
    
    if depth_max is not None:
        # depth_max should be positive (e.g., 500 means show to -500m)
        crop_max = -abs(depth_max)  # Convert to negative
        if crop_max < display_min:  # crop_max is more negative
            display_min = crop_max
            print(f"   🔧 Crop: Maximum depth extended to {abs(crop_max):.0f}m below surface")
            crop_data = True
    
    if height_min is not None:
        # height_min should be positive (e.g., 0 means show from 0m above)
        if height_min > display_max:  # height_min is above current max
            display_max = height_min
            print(f"   🔧 Crop: Minimum height set to {height_min:.0f}m above surface")
            crop_data = True
    
    if height_max is not None:
        # height_max should be positive (e.g., 10 means show to 10m above)
        if height_max < display_max:  # height_max is below current max
            display_max = height_max
            print(f"   🔧 Crop: Maximum height limited to {height_max:.0f}m above surface")
            crop_data = True
    
    # Calculate actual display range
    display_range = display_max - display_min
    
    # Find indices to crop if needed
    if crop_data:
        # Find indices within the crop range
        crop_mask = (surface_relative_vec >= display_min) & (surface_relative_vec <= display_max)
        crop_indices = np.where(crop_mask)[0]
        if len(crop_indices) > 0:
            print(f"   🔧 Data cropping: Keeping {len(crop_indices)}/{len(surface_relative_vec)} depth bins")
        else:
            print(f"   ⚠️  Crop would remove all data, using full range")
            display_min = vec_min
            display_max = vec_max
            crop_indices = None
            crop_data = False
    
    result = {
        'vec_min': vec_min,
        'vec_max': vec_max,
        'total_range': total_range,
        'display_min': display_min,
        'display_max': display_max,
        'display_range': display_range,
        'surface_position': 0.0,
        'max_depth': abs(display_min) if display_min < 0 else 0.0,
        'max_height': display_max if display_max > 0 else 0.0,
        'crop_indices': crop_indices,
        'crop_data': crop_data
    }
    
    # Add topography information if needed
    if use_topo and topo_profile is not None:
        # Convert surface-relative to sea level elevation
        avg_ground_elev = np.mean(topo_profile)
        sea_level_min = avg_ground_elev + display_min
        sea_level_max = avg_ground_elev + display_max
        
        result.update({
            'avg_ground_elev': avg_ground_elev,
            'sea_level_min': sea_level_min,
            'sea_level_max': sea_level_max,
            'ground_min': np.min(topo_profile),
            'ground_max': np.max(topo_profile),
            'topo_variation': np.max(topo_profile) - np.min(topo_profile)
        })
    
    return result

def create_topography_heatmap_with_warping(data_2d, z_vec, topo_profile, distance_m=None,
                                         exaggeration=1.0, log_scale=False, colorscale='Jet',
                                         mode='Energy', title_prefix=""):
    """
    FIXED: Create topography plot using SIMPLER approach
    This creates a heatmap where the Y-axis is warped to follow topography
    
    Parameters:
    -----------
    data_2d : numpy array (depth x width)
        2D tomogram slice (transposed for plotting)
    z_vec : numpy array
        Surface-relative depth vector (0 = surface)
    topo_profile : numpy array
        Topography elevation profile (ALREADY EXAGGERATED if needed)
    distance_m : numpy array, optional
        Actual distance in meters
    exaggeration : float
        Topography vertical exaggeration factor
    log_scale : bool
        Whether to use logarithmic color scaling
    colorscale : str
        Plotly colorscale name
    mode : str
        Data mode ('Energy', 'Velocity', 'Coherence', 'Robustness')
    title_prefix : str
        Title prefix for the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plot with curved topography
    """
    print(f"\n   🏔️  CREATING TOPOGRAPHY HEATMAP")
    print(f"      Data shape: {data_2d.shape} (depth x width)")
    print(f"      z_vec range: {z_vec[0]:.1f}m to {z_vec[-1]:.1f}m (relative to surface)")
    print(f"      Topography range: {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m (sea level)")
    print(f"      Exaggeration factor: {exaggeration}x")
    
    # Create distance array if not provided
    if distance_m is None:
        distance_m = np.arange(data_2d.shape[1])  # Width
        print(f"      Using pixel indices as distance (no real distance)")
    else:
        print(f"      Using real distance: {distance_m[0]:.1f}m to {distance_m[-1]:.1f}m")
    
    # Get dimensions
    num_depths = data_2d.shape[0]  # Depth dimension (rows)
    num_width = data_2d.shape[1]   # Width dimension (columns)
    
    print(f"      Number of depths: {num_depths}")
    print(f"      Number of width points: {num_width}")
    
    # Resample topography to match width if needed
    if len(topo_profile) != num_width:
        print(f"   🔄 Resampling topography: {len(topo_profile)} → {num_width} points")
        topo_profile = np.interp(
            np.linspace(0, 1, num_width),
            np.linspace(0, 1, len(topo_profile)),
            topo_profile
        )
    
    # Create X coordinates (distance along track)
    x_coords = distance_m
    
    # ========== SIMPLER APPROACH: Create regular grid, then warp Y-axis ==========
    # For Heatmap, we need Y coordinates for each row
    # The key insight: We need to create Y coordinates that vary with topography
    
    # Create a base Y grid (surface-relative depths)
    # Then add topography to get true elevation
    
    # For each depth level, create Y coordinates that follow topography
    # elevation = topography + surface_relative_depth
    
    # Create Y coordinates matrix: shape (num_depths, num_width)
    Y_coords = np.zeros((num_depths, num_width))
    for i in range(num_width):
        Y_coords[:, i] = topo_profile[i] + z_vec
    
    print(f"\n   📐 COORDINATE ANALYSIS:")
    print(f"      X coordinates: {len(x_coords)} points")
    print(f"      Y coordinates shape: {Y_coords.shape}")
    print(f"      Y range: {np.min(Y_coords):.1f}m to {np.max(Y_coords):.1f}m")
    print(f"      Ground surface (z_vec=0): {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m")
    
    # ========== DATA PREPARATION ==========
    plot_data = data_2d  # Already in correct orientation (depth x width)
    
    # Apply log scale if requested
    if log_scale and mode in ['Energy', '']:
        print(f"   📊 Applying logarithmic scaling")
        plot_data = np.log10(np.maximum(plot_data, 1e-10))
        valid_data = plot_data[~np.isnan(plot_data)]
        if len(valid_data) > 0:
            z_min = np.percentile(valid_data, 1)  # Avoid -inf
            z_max = np.percentile(valid_data, 99)
        else:
            z_min, z_max = 0, 1
        colorbar_title = 'Log10(Signal Energy)'
    else:
        # Calculate color scaling
        valid_data = plot_data[~np.isnan(plot_data)]
        if len(valid_data) > 0:
            z_min = np.percentile(valid_data, 1)
            z_max = np.percentile(valid_data, 99)
        else:
            z_min, z_max = 0, 1
        
        # Set colorbar title based on mode
        if mode == 'Coherence':
            colorbar_title = 'Coherence'
            z_min, z_max = 0.0, 1.0
        elif mode == 'Velocity':
            colorbar_title = 'Velocity (m/s)'
        elif mode == 'Robustness':
            colorbar_title = 'Robustness Score'
        else:
            colorbar_title = 'Signal Energy'
    
    print(f"   🎨 Color range: {z_min:.3e} to {z_max:.3e}")
    print(f"   🎯 Color scale: {colorscale}")
    
    # ========== CREATE PLOTLY FIGURE ==========
    fig = go.Figure()
    
    # Create hover template
    hover_template = (
        'Distance: %{x:.1f}m<br>' +
        'Elevation: %{y:.1f}m<br>' +
        'Value: %{z:.3e}<extra></extra>'
    )
    
    # ========== FIXED: USE HEATMAP WITH WARPED Y COORDINATES ==========
    # The key is to create a heatmap where each column has its own Y coordinates
    
    print(f"   📊 Creating heatmap with warped Y coordinates...")
    
    # Create a heatmap for the tomogram data
    # We'll use the Y_coords matrix for the Y positions
    # X positions are the same for all depths in a column
    
    # For Heatmap, we need to provide:
    # - z: data values (num_depths x num_width)
    # - x: x coordinates (num_width points)
    # - y: y coordinates for each row (num_depths points)
    
    # However, Heatmap expects y to be 1D (same for all columns)
    # So we need to use the AVERAGE Y for each depth level
    
    # Calculate average Y for each depth level (row)
    y_avg = np.mean(Y_coords, axis=1)
    
    print(f"   📈 Using average Y coordinates for heatmap")
    print(f"      y_avg shape: {y_avg.shape}")
    print(f"      y_avg range: {np.min(y_avg):.1f}m to {np.max(y_avg):.1f}m")
    
    # Create heatmap trace
    heatmap_trace = go.Heatmap(
        z=plot_data.tolist(),
        x=x_coords.tolist(),  # X positions
        y=y_avg.tolist(),     # Y positions (average elevation for each depth)
        colorscale=colorscale,
        zmin=z_min,
        zmax=z_max,
        zsmooth='best',
        name=mode,
        colorbar=dict(
            title=dict(
                text=colorbar_title,
                side='right'
            )
        ),
        hovertemplate=hover_template,
        hoverinfo='x+y+z'
    )
    
    fig.add_trace(heatmap_trace)
    
    # Add topography surface line (ground surface at depth=0)
    surface_elevation = topo_profile  # At depth=0
    
    surface_trace = go.Scatter(
        x=x_coords,
        y=surface_elevation,
        mode='lines',
        line=dict(color='white', width=3),
        name='Ground Surface',
        hoverinfo='x+y+name',
        hovertemplate='Distance: %{x:.1f}m<br>Elevation: %{y:.1f}m<br>Ground Surface<extra></extra>'
    )
    
    fig.add_trace(surface_trace)
    
    # Add sea level line (0m elevation) - only if within view range
    sea_level = 0.0
    y_min_global = np.min(Y_coords)
    y_max_global = np.max(Y_coords)
    
    if y_min_global <= sea_level <= y_max_global:
        sea_level_line = go.Scatter(
            x=[x_coords[0], x_coords[-1]],
            y=[sea_level, sea_level],
            mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            name='Sea Level (0m)',
            hoverinfo='x+y+name',
            hovertemplate='Distance: %{x:.1f}m<br>Elevation: %{y:.1f}m<br>Sea Level<extra></extra>'
        )
        fig.add_trace(sea_level_line)
    
    # ========== CONFIGURE LAYOUT ==========
    # Calculate appropriate Y-axis range
    # Use the full range of Y coordinates
    y_min = y_min_global - 10  # Small padding
    y_max = y_max_global + 10  # Small padding
    
    print(f"\n   📊 Y-AXIS CONFIGURATION:")
    print(f"      Tomogram elevation range: {y_min_global:.1f}m to {y_max_global:.1f}m")
    print(f"      Ground surface range: {np.min(surface_elevation):.1f}m to {np.max(surface_elevation):.1f}m")
    print(f"      Sea level: {sea_level}m")
    print(f"      Final Y-axis: {y_min:.1f}m to {y_max:.1f}m")
    print(f"      Total vertical span: {y_max - y_min:.1f}m")
    
    # Update layout with correct title based on exaggeration
    if exaggeration == 1.0:
        exaggeration_text = "No exaggeration"
        title_suffix = ""
    else:
        exaggeration_text = f"{exaggeration}x exaggeration"
        title_suffix = f" ({exaggeration_text})"
    
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} - Topographic Display{title_suffix}",
            font=dict(size=16, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title='Distance Along Track (m)',
            title_font=dict(size=14),
            gridcolor='rgba(255,255,255,0.1)',
            range=[x_coords[0], x_coords[-1]]
        ),
        yaxis=dict(
            title='Elevation Above Sea Level (m)',
            title_font=dict(size=14),
            gridcolor='rgba(255,255,255,0.1)',
            range=[y_min, y_max]
        ),
        template='plotly_dark',
        height=700,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1.0,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        margin=dict(l=60, r=120, t=60, b=60)
    )
    
    print(f"\n   ✅ TOPOGRAPHY HEATMAP CREATED")
    print(f"      Exaggeration: {exaggeration_text}")
    print(f"      Y-axis range: {y_min:.1f}m to {y_max:.1f}m")
    print(f"      Data points: {num_depths * num_width:,}")
    print(f"      Title: {title_prefix} - Topographic Display{title_suffix}")
    
    return fig

@app.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    try:
        req = request.json
        mode = req.get('mode', 'Energy') 
        slice_type = req.get('slice_type', 'Y') 
        slice_index = int(req.get('slice_index', 0))
        use_topo = bool(req.get('use_topo', False))
        topo_exaggeration = float(req.get('topo_exaggeration', 1.0))  # Default to 1.0x (no exaggeration)
        log_scale = bool(req.get('log_scale', False))
        
        # Optional crop parameters
        depth_min = req.get('depth_min')
        depth_max = req.get('depth_max')
        height_min = req.get('height_min')
        height_max = req.get('height_max')
        
        # --- DEBUG PRINTS ---
        print(f"\n" + "="*70)
        print(f"🚀 DEBUG PLOT REQUEST")
        print("="*70)
        print(f"   Mode: {mode}")
        print(f"   Slice Type: {slice_type}")
        print(f"   Slice Index: {slice_index}")
        print(f"   Use Topography: {use_topo}")
        print(f"   Topo Exaggeration: {topo_exaggeration}x")
        print(f"   Log Scale: {log_scale}")
        if depth_min is not None or depth_max is not None or height_min is not None or height_max is not None:
            print(f"   Crop - Depth: [{depth_min}, {depth_max}], Height: [{height_min}, {height_max}]")
        print("="*70)

        # 1. VALIDATE DATA EXISTS
        if GLOBAL_DATA is None:
            print("❌ ERROR: No data loaded in GLOBAL_DATA")
            return jsonify({'error': 'No data loaded'})
        
        if 'tomogram_cube' not in GLOBAL_DATA:
            print("❌ ERROR: No tomogram_cube in data")
            return jsonify({'error': 'No tomogram data available'})
        
        # 2. GET BASE VOLUME
        base_vol = None
        if mode == 'Velocity' and 'velocity_cube' in GLOBAL_DATA:
            base_vol = GLOBAL_DATA['velocity_cube']
            print(f"   Using velocity cube")
        elif mode == 'Coherence' and 'coherence_cube' in GLOBAL_DATA:
            base_vol = GLOBAL_DATA['coherence_cube']
            print(f"   Using coherence cube")
        elif mode == 'Robustness' and 'robustness_score_3d' in GLOBAL_DATA:
            base_vol = GLOBAL_DATA['robustness_score_3d']
            print(f"   Using robustness cube")
        else:
            base_vol = GLOBAL_DATA['tomogram_cube']
            print(f"   Using tomogram cube for {mode} mode")
        
        if base_vol is None:
            print("⚠️  Requested data not available, falling back to tomogram")
            base_vol = GLOBAL_DATA['tomogram_cube']
        
        # Convert complex to magnitude if needed
        if np.iscomplexobj(base_vol):
            print(f"   Converting complex data to magnitude")
            base_vol = np.abs(base_vol)
        
        # 3. EXTRACT SLICE
        print(f"   Cube shape: {base_vol.shape}")
        
        # Validate slice index
        if slice_type == 'X':
            max_idx = base_vol.shape[0] - 1
            if slice_index > max_idx:
                print(f"⚠️  Slice index {slice_index} > max {max_idx}, clamping")
                slice_index = max_idx
            raw_slice = base_vol[slice_index, :, :].T  # Transpose for display
            title_prefix = f"Cross-Track (Row {slice_index})"
            x_lbl = "Range (Cross-Track)"
            y_lbl = "Surface-Relative (m)"
            use_topo = False  # Topography doesn't apply to X-slices
            
        elif slice_type == 'Y':
            max_idx = base_vol.shape[1] - 1
            if slice_index > max_idx:
                print(f"⚠️  Slice index {slice_index} > max {max_idx}, clamping")
                slice_index = max_idx
            raw_slice = base_vol[:, slice_index, :].T  # Transpose for display
            title_prefix = f"Along-Track (Col {slice_index})"
            x_lbl = "Azimuth (Distance)"
            y_lbl = "Surface-Relative (m)"
            
        elif slice_type == 'Z':
            max_idx = base_vol.shape[2] - 1
            if slice_index > max_idx:
                print(f"⚠️  Slice index {slice_index} > max {max_idx}, clamping")
                slice_index = max_idx
            raw_slice = base_vol[:, :, slice_index].T  # Transpose for display
            title_prefix = f"Depth Slice {slice_index}"
            x_lbl = "Range (Cols)"
            y_lbl = "Azimuth"
            use_topo = False  # Topography doesn't apply to depth slices
            
        else:
            print(f"❌ ERROR: Invalid slice type: {slice_type}")
            return jsonify({'error': f'Invalid slice type: {slice_type}'})
        
        # Handle NaN values
        heatmap_data = np.nan_to_num(raw_slice, nan=0.0)
        h, w = heatmap_data.shape
        print(f"   📐 Slice shape: {w} (width) x {h} (height)")
        print(f"   📊 Data range: {np.min(heatmap_data):.3e} to {np.max(heatmap_data):.3e}")
        
        # 4. GET CORRECTED SURFACE-RELATIVE VECTOR
        print(f"\n   🔍 GETTING SURFACE-RELATIVE VECTOR:")
        surface_relative_vec = get_surface_relative_vector(h)
        
        # 5. TOPOGRAPHY HANDLING
        has_topography = False
        topo_profile = None
        display_info = {}
        
        if use_topo and slice_type == 'Y':
            print(f"\n   🌍 ANALYZING TOPOGRAPHY:")
            
            # Check for topography data
            has_online = GLOBAL_DATA.get('has_online_topo', False)
            has_tiff = GLOBAL_DATA.get('has_tiff_topo', False)
            has_flat = GLOBAL_DATA.get('flat_topo', False)
            
            if has_online:
                topo_profile = GLOBAL_DATA.get('online_elevation_profile')
                topo_source = "OpenTopoData"
            elif has_tiff:
                topo_profile = GLOBAL_DATA.get('tiff_elevation_profile')
                topo_source = "TIFF"
            elif has_flat:
                topo_profile = GLOBAL_DATA.get('elevation_profile', None)
                topo_source = "Flat"
            else:
                topo_profile = GLOBAL_DATA.get('elevation_profile', None)
                topo_source = "Generic"
            
            if topo_profile is not None and len(topo_profile) > 0:
                print(f"   📍 Topography from {topo_source}: {len(topo_profile)} points")
                print(f"   📈 Raw elevation range: {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m")
                
                # Check if topography has variation
                topo_std = np.std(topo_profile)
                print(f"   📊 Topography std dev: {topo_std:.2f}m")
                
                if topo_std < 0.5:
                    print(f"   ⚠️  WARNING: Very little topography variation")
                
                # Resample if needed
                if len(topo_profile) != w:
                    print(f"   🔄 Resampling topography to match width {w}")
                    topo_profile = resample_profile_to_width(topo_profile, w)
                
                # ========== FIXED: APPLY EXAGGERATION ONLY IF NEEDED ==========
                # Store original topography for display info
                original_topo_profile = topo_profile.copy()
                
                # Apply vertical exaggeration only if not 1.0
                if topo_exaggeration != 1.0:
                    print(f"   🔼 Applying {topo_exaggeration}x vertical exaggeration")
                    topo_profile = add_vertical_exaggeration(topo_profile, topo_exaggeration, method='range')
                    print(f"   📊 Exaggerated topography range: {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m")
                else:
                    print(f"   🔼 No vertical exaggeration (1x)")
                    # Ensure we're using the original topography
                    topo_profile = original_topo_profile
                
                # Calculate display range with ORIGINAL topography for proper scaling
                display_info = calculate_vertical_range(
                    surface_relative_vec, use_topo, original_topo_profile,
                    depth_min, depth_max, height_min, height_max
                )
                
                # Get distance in meters
                if 'total_dist_m' in GLOBAL_DATA and 'pixel_spacing_m' in GLOBAL_DATA:
                    total_dist = GLOBAL_DATA['total_dist_m']
                    pixel_spacing = GLOBAL_DATA['pixel_spacing_m']
                    distance_m = np.arange(w) * pixel_spacing
                    print(f"   📏 Real distance: {distance_m[0]:.1f}m to {distance_m[-1]:.1f}m")
                else:
                    distance_m = np.arange(w)
                    print(f"   📏 Using pixel indices as distance")
                
                # ========== USE CORRECTED TOPOGRAPHY PLOT FUNCTION ==========
                # Create topography plot using the corrected function
                fig = create_topography_heatmap_with_warping(
                    data_2d=heatmap_data,
                    z_vec=surface_relative_vec,
                    topo_profile=topo_profile,  # Already exaggerated if needed
                    distance_m=distance_m,
                    exaggeration=topo_exaggeration,  # Pass for display only
                    log_scale=log_scale,
                    colorscale='Jet' if mode == 'Energy' else 'Viridis',
                    mode=mode,
                    title_prefix=title_prefix
                )
                
                has_topography = True
                print(f"\n   ✅ TOPOGRAPHY PLOT CREATED")
                
            else:
                print(f"⚠️  Topography requested but no profile available")
                use_topo = False
        
        # 6. APPLY CROPPING IF NEEDED
        if not has_topography:
            if display_info.get('crop_data', False) and display_info.get('crop_indices') is not None:
                crop_idx = display_info['crop_indices']
                if len(crop_idx) > 0:
                    print(f"   ✂️  Cropping tomogram data: {h} depth bins → {len(crop_idx)} depth bins")
                    # Crop the heatmap data
                    heatmap_data = heatmap_data[crop_idx, :]
                    # Crop the surface-relative vector
                    surface_relative_vec = surface_relative_vec[crop_idx]
                    h = len(crop_idx)  # Update height
                    
                    print(f"   📏 Cropped elevation range: {surface_relative_vec[0]:.1f}m to {surface_relative_vec[-1]:.1f}m")
                    print(f"   📊 Cropped data shape: {heatmap_data.shape}")
        
        # 7. DATA SCALING AND COLOR (for non-topography plots)
        if not has_topography:
            valid_data = heatmap_data[heatmap_data != 0]
            
            # Set color scale based on mode
            if mode == 'Coherence':
                colorscale = 'Hot'
                z_min, z_max = 0.0, 1.0
                colorbar_title = 'Coherence'
            elif mode == 'Velocity':
                colorscale = 'Viridis'
                if valid_data.size > 0:
                    z_min = np.min(valid_data)
                    z_max = np.max(valid_data)
                else:
                    z_min, z_max = 0.0, 1.0
                colorbar_title = 'Velocity (m/s)'
            elif mode == 'Robustness':
                colorscale = 'RdYlGn'
                if valid_data.size > 0:
                    z_min = np.min(valid_data)
                    z_max = np.max(valid_data)
                else:
                    z_min, z_max = 0.0, 10.0
                colorbar_title = 'Robustness Score'
            else:  # Energy or default
                colorscale = 'Jet'
                if valid_data.size > 0:
                    z_min = np.min(valid_data)
                    # Use percentile to avoid outliers
                    z_max = np.percentile(valid_data, 99.5)
                else:
                    z_min, z_max = 0.0, 1.0
                colorbar_title = 'Signal Energy'
            
            # Apply log scale if requested
            if log_scale and mode in ['Energy', '']:
                print(f"   Applying logarithmic scaling")
                heatmap_data = np.log10(np.maximum(heatmap_data, 1e-10))
                if z_min > 0:
                    z_min = np.log10(z_min)
                if z_max > 0:
                    z_max = np.log10(z_max)
                colorbar_title = 'Log10(Signal Energy)'
            
            print(f"\n   🎨 COLOR SCALING:")
            print(f"      Display range: {z_min:.3e} to {z_max:.3e}")
            print(f"      Color scale: {colorscale}")
            
            # 8. CREATE REGULAR (NON-TOPOGRAPHY) PLOT
            fig = go.Figure()
            
            # Create hover template
            hover_template = (
                'Distance: %{x}<br>' +
                'Surface-Relative: %{y:.1f}m<br>' +
                'Value: %{z:.3e}<extra></extra>'
            )
            
            # For non-topography: use regular heatmap
            y_coords = surface_relative_vec.tolist()
            
            heatmap_trace = go.Heatmap(
                z=heatmap_data.tolist(),
                x=list(range(w)),  # X coordinates (distance indices)
                y=y_coords,  # Y coordinates (elevation values)
                colorscale=colorscale,
                zmin=z_min,
                zmax=z_max,
                zsmooth='best',
                name=mode,
                colorbar=dict(
                    title=dict(
                        text=colorbar_title,
                        side='right'
                    )
                ),
                hovertemplate=hover_template
            )
            
            fig.add_trace(heatmap_trace)
            
            # Update layout for non-topography plot
            layout_updates = {
                'title': dict(
                    text=title_prefix,
                    font=dict(size=16, color='white'),
                    x=0.5
                ),
                'xaxis': dict(
                    title=x_lbl,
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                'yaxis': dict(
                    title=y_lbl,
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                'template': "plotly_dark",
                'margin': dict(l=60, r=30, t=60, b=60),
                'height': 700,
                'showlegend': False
            }
            
            fig.update_layout(**layout_updates)
        
        print(f"\n   ✅ Plot created successfully")
        print(f"   📤 Sending response...")
        print("="*70)
        
        # Return the plot as JSON
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return app.response_class(
            response=fig_json,
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        print(f"\n❌ SERVER ERROR in get_plot_data: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/get_metadata')
def get_metadata():
    """Return metadata about the loaded dataset."""
    if GLOBAL_DATA is None:
        return jsonify({'error': 'No data loaded'})
    
    try:
        # Get basic metadata
        metadata = {
            'filename': GLOBAL_DATA.get('filename_npz', 'Unknown'),
            'dims': list(GLOBAL_DATA['tomogram_cube'].shape) if 'tomogram_cube' in GLOBAL_DATA else [],
            'has_online_topo': GLOBAL_DATA.get('has_online_topo', False),
            'has_tiff_topo': GLOBAL_DATA.get('has_tiff_topo', False),
            'has_any_topo': GLOBAL_DATA.get('has_online_topo', False) or GLOBAL_DATA.get('has_tiff_topo', False),
            'tiff_filename': GLOBAL_DATA.get('tiff_filename', None),
            'flat_topo': GLOBAL_DATA.get('flat_topo', False)
        }
        
        # Add surface-relative information if available
        if 'z_vec' in GLOBAL_DATA and GLOBAL_DATA['z_vec'] is not None:
            z_vec = GLOBAL_DATA['z_vec']
            metadata['surface_relative_range'] = [float(z_vec[0]), float(z_vec[-1])]
            metadata['surface_relative_resolution'] = float(z_vec[1] - z_vec[0]) if len(z_vec) > 1 else 0.0
            
            # Calculate depth and height ranges
            below_surface = z_vec[z_vec < 0]
            above_surface = z_vec[z_vec > 0]
            
            if len(below_surface) > 0:
                metadata['max_depth'] = float(abs(np.min(below_surface)))
            else:
                metadata['max_depth'] = 0.0
                
            if len(above_surface) > 0:
                metadata['max_height'] = float(np.max(above_surface))
            else:
                metadata['max_height'] = 0.0
            
            metadata['surface_position'] = 0.0
            metadata['interpretation'] = '0m = Ground surface, Negative = Below surface, Positive = Above surface'
        
        print(f"📋 Metadata request - returning: {metadata}")
        return jsonify(metadata)
        
    except Exception as e:
        print(f"❌ Error in get_metadata: {e}")
        return jsonify({'error': str(e)})

@app.route('/get_topo_options')
def get_topo_options():
    """Return topography options."""
    if GLOBAL_DATA is None:
        return jsonify({'error': 'No data loaded'})
    
    try:
        # Check what topography is available
        has_online = GLOBAL_DATA.get('has_online_topo', False)
        has_tiff = GLOBAL_DATA.get('has_tiff_topo', False)
        has_any = has_online or has_tiff
        
        options = {
            'has_online_topo': has_online,
            'has_tiff_topo': has_tiff,
            'has_any_topo': has_any,
            'tiff_filename': GLOBAL_DATA.get('tiff_filename', None),
            'default_exaggeration': 1.0,  # CHANGED: Default to 1.0x (no exaggeration)
            'available_exaggerations': [1, 2, 3, 5, 10, 20]
        }
        
        # Add elevation range if available
        if has_online and 'online_elevation_profile' in GLOBAL_DATA:
            profile = GLOBAL_DATA['online_elevation_profile']
            if len(profile) > 0:
                options['elevation_range'] = {
                    'min': float(np.min(profile)),
                    'max': float(np.max(profile)),
                    'mean': float(np.mean(profile))
                }
        elif has_tiff and 'tiff_elevation_profile' in GLOBAL_DATA:
            profile = GLOBAL_DATA['tiff_elevation_profile']
            if len(profile) > 0:
                options['elevation_range'] = {
                    'min': float(np.min(profile)),
                    'max': float(np.max(profile)),
                    'mean': float(np.mean(profile))
                }
        
        print(f"📋 Topography options: {options}")
        return jsonify(options)
        
    except Exception as e:
        print(f"❌ Error in get_topo_options: {e}")
        return jsonify({'error': str(e)})

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify server is working."""
    return jsonify({
        'status': 'ok',
        'message': 'PAR-Capella server is running',
        'data_loaded': GLOBAL_DATA is not None,
        'timestamp': datetime.now().isoformat()
    })

# Serve static files
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('.', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    print("Starting PAR-Capella server...")
    # This is for direct execution, but normally started via start_flask_server()
    app.run(debug=True, port=5000)