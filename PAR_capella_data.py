# PAR_capella_data.py
import numpy as np
import os

def load_tomography_data(filepath):
    try:
        with np.load(filepath, allow_pickle=True) as data:
            # Load ALL keys from the NPZ file, not just specific ones
            container = {k: data[k] for k in data.files}
            
        # ========== ADDED DEBUG INFORMATION ==========
        print(f"\n   🔍 LOADED DATA DEBUG:")
        print(f"      File: {os.path.basename(filepath)}")
        print(f"      Total keys loaded: {len(container)}")
        print(f"      Keys: {list(container.keys())}")
        
        # Check for required keys but don't stop loading if some are missing
        required_keys = ['tomogram_cube', 'z_vec', 'radar_params']
        for key in required_keys:
            if key not in container:
                print(f"⚠️ Warning: File {filepath} missing key '{key}'")

        # Debug: Show what keys were loaded
        print(f"   ✓ Loaded: {os.path.basename(filepath)}")
        print(f"      Shape: {container.get('tomogram_cube', 'Empty').shape}")
        
        # Specifically check for geographic coordinates
        if 'geo_start' in container and 'geo_end' in container:
            geo_start = container['geo_start']
            geo_end = container['geo_end']
            
            # ========== ADDED: DETAILED GEOGRAPHIC DEBUG ==========
            print(f"\n   🔍 GEOGRAPHIC COORDINATES DEBUG:")
            print(f"      geo_start: {geo_start}")
            print(f"      geo_end: {geo_end}")
            
            # Check if coordinates are identical
            if geo_start is not None and geo_end is not None:
                if len(geo_start) >= 2 and len(geo_end) >= 2:
                    print(f"      Start: ({geo_start[0]:.6f}°, {geo_start[1]:.6f}°)")
                    print(f"      End:   ({geo_end[0]:.6f}°, {geo_end[1]:.6f}°)")
                    
                    # Calculate distance
                    from PAR_capella_topography import haversine_distance
                    dist = haversine_distance(geo_start[0], geo_start[1], geo_end[0], geo_end[1])
                    print(f"      Distance: {dist:.2f} meters")
                    
                    if geo_start[0] == geo_end[0] and geo_start[1] == geo_end[1]:
                        print(f"      ⚠️  CRITICAL: Start and end coordinates are IDENTICAL!")
                        print(f"         This will cause topography extraction issues")
                        print(f"         Expected different coordinates for a line segment")
                    else:
                        print(f"      ✅ Coordinates are different (good for line)")
                        
                    # Check if it's a valid line (more than just a point)
                    if dist < 10.0:  # Less than 10 meters
                        print(f"      ⚠️  WARNING: Very short line ({dist:.2f}m)")
                        print(f"         Expected longer line for tomographic section")
                else:
                    print(f"      ⚠️  Invalid coordinate format")
            else:
                print(f"      ⚠️  One or both coordinates are None")
        else:
            print(f"      ⚠️  No geographic coordinates found in file")
            
        # ========== ADDED: CHECK FOR OTHER IMPORTANT FIELDS ==========
        important_fields = ['tomogram_cube', 'coherence_cube', 'velocity_map_3d', 
                           'robustness_score_3d', 'z_vec', 'processed_column_indices']
        
        print(f"\n   🔍 IMPORTANT FIELD CHECK:")
        for field in important_fields:
            if field in container:
                value = container[field]
                if hasattr(value, 'shape'):
                    print(f"      {field}: {value.shape} {value.dtype}")
                else:
                    print(f"      {field}: {type(value)}")
            else:
                print(f"      {field}: ❌ MISSING")
        
        # Check z_vec orientation
        if 'z_vec' in container:
            z_vec = container['z_vec']
            if z_vec is not None and len(z_vec) > 1:
                print(f"\n   🔍 Z_VECTOR ANALYSIS:")
                print(f"      Shape: {z_vec.shape}, Type: {z_vec.dtype}")
                print(f"      Range: {z_vec[0]:.1f}m to {z_vec[-1]:.1f}m")
                print(f"      Resolution: {abs(z_vec[1] - z_vec[0]):.2f}m per bin")
                print(f"      Total depth: {abs(z_vec[-1] - z_vec[0]):.1f}m")
                if z_vec[0] < z_vec[-1]:
                    print(f"      Orientation: Increases upward (negative → positive)")
                    print(f"      Interpretation: 0m = surface, negative = depth, positive = height")
                else:
                    print(f"      Orientation: Decreases upward (positive → negative)")
                    print(f"      ⚠️  May need reversal")
        
        # ========== NEW: COMPREHENSIVE TOPOGRAPHY DATA AUDIT ==========
        print(f"\n" + "="*70)
        print(f"   🔍 TOPOGRAPHY DATA AUDIT")
        print("="*70)
        
        # Check for any topography-related fields
        topo_keys = [k for k in container.keys() if any(topo_term in k.lower() 
                    for topo_term in ['elev', 'topo', 'height', 'dem', 'surface', 'ground', 'altitude'])]
        
        print(f"\n   1. TOPOGRAPHY-RELATED KEYS IN NPZ FILE:")
        if topo_keys:
            for key in topo_keys:
                value = container[key]
                if hasattr(value, 'shape'):
                    print(f"      • {key}:")
                    print(f"        Shape: {value.shape}, Type: {value.dtype}")
                    if len(value.shape) == 1:
                        if len(value) > 0:
                            print(f"        Length: {len(value)} points")
                            print(f"        Range: {np.min(value):.1f}m to {np.max(value):.1f}m")
                            print(f"        Mean: {np.mean(value):.1f}m, Std: {np.std(value):.1f}m")
                            
                            # Check if it's constant (flat)
                            if np.std(value) < 0.1:
                                print(f"        ⚠️  CONSTANT ELEVATION (likely flat or placeholder)")
                            else:
                                variation = np.max(value) - np.min(value)
                                print(f"        ✅ Varying topography ({variation:.1f}m range)")
                                
                            # Show first and last few values
                            if len(value) <= 10:
                                print(f"        Values: {value}")
                            else:
                                print(f"        First 5: {value[:5]}")
                                print(f"        Last 5: {value[-5:]}")
                    else:
                        print(f"        Multi-dimensional array - not a topography profile")
                else:
                    # Scalar or non-array value
                    print(f"      • {key}: {type(value)} = {value}")
        else:
            print(f"      ⚠️  NO topography-related keys found in NPZ file")
        
        # Check specific expected topography field names
        expected_topo_fields = ['elevation_profile', 'topo_profile', 'dem_elevation', 
                               'surface_elevation', 'ground_elevation', 'elevation_data',
                               'topography_profile', 'height_profile']
        
        print(f"\n   2. CHECKING EXPECTED TOPOGRAPHY FIELD NAMES:")
        for field in expected_topo_fields:
            if field in container:
                val = container[field]
                print(f"      • Found '{field}':")
                if hasattr(val, '__len__'):
                    print(f"        Length: {len(val)} points")
                    if len(val) > 0:
                        print(f"        First 10 values: {val[:10]}")
                        print(f"        Min: {np.min(val):.1f}m, Max: {np.max(val):.1f}m")
                        print(f"        Std dev: {np.std(val):.2f}m")
                        
                        # Check for suspicious patterns
                        unique_vals = np.unique(val.round(1))
                        if len(unique_vals) <= 3:
                            print(f"        ⚠️  SUSPICIOUS: Only {len(unique_vals)} unique values")
                            print(f"        Unique values: {unique_vals}")
                else:
                    print(f"        Type: {type(val)}, Value: {val}")
        
        # Check z-vector for topography compatibility
        print(f"\n   3. Z-VECTOR ANALYSIS FOR TOPOGRAPHY WARPING:")
        if 'z_vec' in container and container['z_vec'] is not None:
            z_vec = container['z_vec']
            print(f"      • Vector length: {len(z_vec)}")
            print(f"      • Monotonic: {np.all(np.diff(z_vec) > 0) or np.all(np.diff(z_vec) < 0)}")
            print(f"      • Direction: {'Increasing' if z_vec[0] < z_vec[-1] else 'Decreasing'}")
            print(f"      • Surface position in vector:")
            
            # Find where surface (0m) would be
            if z_vec[0] <= 0 <= z_vec[-1] or z_vec[-1] <= 0 <= z_vec[0]:
                # Surface is within vector range
                surface_idx = np.argmin(np.abs(z_vec))
                print(f"        Surface (0m) near index {surface_idx}: {z_vec[surface_idx]:.2f}m")
                print(f"        {surface_idx}/{len(z_vec)} bins from top")
            else:
                print(f"        ⚠️  Surface (0m) NOT in vector range")
                print(f"        Vector range: {z_vec[0]:.2f}m to {z_vec[-1]:.2f}m")
            
            # Count bins above and below surface
            if len(z_vec) > 0:
                above_surface = np.sum(z_vec > 0)
                below_surface = np.sum(z_vec < 0)
                at_surface = np.sum(z_vec == 0)
                print(f"        Bins above surface (>0m): {above_surface}")
                print(f"        Bins below surface (<0m): {below_surface}")
                print(f"        Bins at surface (0m): {at_surface}")
        
        # Check radar parameters for elevation info
        print(f"\n   4. RADAR PARAMETERS RELATED TO ELEVATION:")
        if 'radar_params' in container:
            radar_params = container['radar_params']
            if isinstance(radar_params, dict):
                for key in ['height', 'elevation', 'altitude', 'surface']:
                    if key in radar_params:
                        print(f"      • radar_params['{key}']: {radar_params[key]}")
            else:
                print(f"      • radar_params type: {type(radar_params)}")
        
        # Final assessment
        print(f"\n   5. TOPOGRAPHY DATA ASSESSMENT:")
        
        # Count how many topography profiles we have
        topo_profiles = []
        for key in container.keys():
            val = container[key]
            if hasattr(val, 'shape') and len(val.shape) == 1:
                if len(val) > 10:  # Reasonable length for topography
                    # Check if it could be elevation data (typical ranges)
                    if np.all(np.abs(val) < 10000):  # Reasonable elevation range
                        topo_profiles.append(key)
        
        print(f"      • Potential topography profiles: {len(topo_profiles)}")
        if topo_profiles:
            print(f"      • Profile names: {topo_profiles}")
            print(f"      • These will be used for external topography fetching if needed")
        else:
            print(f"      ⚠️  No suitable topography profiles found in NPZ")
            print(f"      • External topography (TIFF/OpenTopoData) will be required")
        
        print("="*70)
        
        return container
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def merge_datasets(file_list):
    """
    Loads multiple .npz files, sorts them by column index,
    and stitches the cubes together along Axis 1.
    """
    loaded_datasets = []
    
    for fname in file_list:
        d = load_tomography_data(fname)
        if d is not None:
            start_col = d['processed_column_indices'][0]
            loaded_datasets.append({'data': d, 'start_col': start_col})
    
    if not loaded_datasets:
        return None

    loaded_datasets.sort(key=lambda x: x['start_col'])
    print(f"   ➤ Sorting {len(loaded_datasets)} file chunks...")

    reference = loaded_datasets[0]['data']
    ref_z = reference['z_vec']
    ref_shape = reference['tomogram_cube'].shape
    
    tomo_list = []
    vel_list = []
    coh_list = []
    col_indices_list = []
    
    for item in loaded_datasets:
        d = item['data']
        tomo_list.append(d['tomogram_cube'])
        vel_list.append(d.get('velocity_cube', np.zeros_like(d['tomogram_cube'])))
        coh_list.append(d.get('coherence_cube', np.zeros_like(d['tomogram_cube'])))
        col_indices_list.append(d['processed_column_indices'])

    print("   ➤ Stitching data cubes...")
    merged_tomo = np.concatenate(tomo_list, axis=1)
    merged_vel = np.concatenate(vel_list, axis=1)
    merged_coh = np.concatenate(coh_list, axis=1)
    merged_cols = np.concatenate(col_indices_list)

    final_container = reference.copy()
    final_container['tomogram_cube'] = merged_tomo
    final_container['velocity_cube'] = merged_vel
    final_container['coherence_cube'] = merged_coh
    final_container['processed_column_indices'] = merged_cols
    
    last_data = loaded_datasets[-1]['data']
    if 'geo_end' in last_data:
        final_container['geo_end'] = last_data['geo_end']

    print(f"   ✅ Merge Complete. New Shape: {merged_tomo.shape}")
    return final_container