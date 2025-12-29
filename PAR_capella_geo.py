# PAR_capella_geo.py
import numpy as np
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def integrate_topography_and_coordinates(container):
    """
    Basic geographic integration - now simpler since topography is handled elsewhere.
    
    Parameters:
    -----------
    container : dict
        Data container with tomogram cube and optional geo coordinates
    
    Returns:
    --------
    container : dict
        Updated container with distance information
    """
    print("🌍 [PAR-Geo] Integrating geographic coordinates...")
    
    geo_start = container.get('geo_start')
    geo_end = container.get('geo_end')
    
    # Get tomogram dimensions
    if 'tomogram_cube' in container:
        axis_len = float(container['tomogram_cube'].shape[0])
    else:
        axis_len = 1000  # Default fallback
        print("   ⚠️ No tomogram cube found, using default length")
    
    # Calculate or estimate total distance
    if geo_start is None or geo_end is None:
        total_dist = axis_len * 2.0  # Assume 2m/pixel spacing
        print(f"   ⚠️ No geographic coordinates, using estimated distance: {total_dist:.1f} m")
    else:
        # Calculate actual geodesic distance
        total_dist = haversine_distance(geo_start[0], geo_start[1], geo_end[0], geo_end[1])
        
        # Safety check for very short or zero distances
        if total_dist < 1.0:
            print(f"   ⚠️ Very short distance ({total_dist:.2f} m), using pixel-based estimate")
            total_dist = axis_len * 2.0  # Assume 2m/pixel spacing
        
        print(f"   📏 Geodesic distance: {total_dist:.1f} m")
    
    # Store distance information
    container['total_dist_m'] = float(total_dist)
    
    # Calculate pixel spacing (meters per pixel)
    if axis_len > 1:
        pixel_spacing = total_dist / (axis_len - 1)
        container['pixel_spacing_m'] = float(pixel_spacing)
        print(f"   📐 Pixel spacing: {pixel_spacing:.2f} m/px")
    else:
        container['pixel_spacing_m'] = 2.0  # Default
        print(f"   📐 Using default pixel spacing: 2.0 m/px")
    
    # Set basic flags (topography will be set by main.py)
    container['has_tiff_topo'] = False
    container['has_online_topo'] = False
    container['tiff_filename'] = "None"
    container['flat_topo'] = False
    
    print("   ✅ Geographic integration complete")
    
    return container