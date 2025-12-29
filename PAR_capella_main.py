# PAR_capella_main.py - FIXED: Correct topography handling with downsampling options
import os
import sys
import webbrowser
import threading
import numpy as np
import glob
from scipy.ndimage import zoom
from datetime import datetime

# IMPORT MODULES
from PAR_capella_data import load_tomography_data, merge_datasets
from PAR_capella_geo import integrate_topography_and_coordinates
from PAR_capella_server import start_flask_server
from PAR_capella_topography import (
    get_topography_from_opentopodata,
    get_topography_from_tiff,
    HAS_RASTERIO,
    validate_topography_data,
    create_flat_topography
)

def print_banner():
    """Print the application banner."""
    print("=" * 70)
    print("   PAR-CAPELLA: ADVANCED GEOGRAPHIC TOMOGRAPHY VIEWER")
    print("   Version 2.0 - AN-style Topography Integration")
    print("=" * 70)
    print("   Features:")
    print("   • 3D Tomogram Visualization")
    print("   • AN-style Topography Display")
    print("   • TIFF & OpenTopoData Elevation Support")
    print("   • Multiple Display Modes (Energy, Velocity, Coherence)")
    print("=" * 70)

def select_file_interactive(files):
    """Interactive file selection with details."""
    print(f"\n📂 Found {len(files)} tomography datasets:")
    print("-" * 60)
    
    for i, f in enumerate(files):
        file_path = os.path.join(os.getcwd(), f)
        try:
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            file_time = os.path.getctime(file_path)
            time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-d %H:%M')
            
            # Try to get dimensions
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'tomogram_cube' in data:
                    cube = data['tomogram_cube']
                    dims = f"{cube.shape[0]}×{cube.shape[1]}×{cube.shape[2]}"
                else:
                    dims = "Unknown"
                data.close()
            except:
                dims = "Unknown"
            
            print(f"   [{i+1}] {f}")
            print(f"       Size: {file_size:.1f} MB | Dimensions: {dims}")
            print(f"       Created: {time_str}")
            
            if i < len(files) - 1:
                print()
                
        except Exception as e:
            print(f"   [{i+1}] {f} (Error reading: {e})")
    
    print("-" * 60)
    
    while True:
        try:
            choice = input(f"\n   Select dataset (1-{len(files)}, or Enter for newest): ").strip()
            if choice == '':
                return 0  # Newest file
            else:
                sel = int(choice) - 1
                if 0 <= sel < len(files):
                    return sel
                else:
                    print(f"   ❌ Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("   ❌ Please enter a valid number")

def main():
    # Print banner
    print_banner()
    
    cwd = os.getcwd()
    
    # Find all NPZ files
    all_npz_files = glob.glob(os.path.join(cwd, "tomography_3D_results*.npz"))
    files = []
    
    for f in all_npz_files:
        fname = os.path.basename(f)
        # Skip CV5_OPTIMIZED files
        if '_CV5_OPTIMIZED' not in fname:
            files.append(fname)
    
    # Sort by creation time (newest first)
    files.sort(key=lambda x: os.path.getctime(os.path.join(cwd, x)), reverse=True)
    
    if not files:
        print("\n❌ ERROR: No tomography result files found!")
        print("   Please ensure you have run the 3D processor first.")
        print("   Expected files: tomography_3D_results_*.npz")
        print("   Current directory:", cwd)
        print("   Files found:", os.listdir(cwd))
        input("\n   Press Enter to exit...")
        return
    
    # Select file
    selected_idx = select_file_interactive(files)
    selected_file = files[selected_idx]
    print(f"\n✓ Selected: {selected_file}")
    
    # Load Data
    print(f"\n📂 Loading dataset...")
    data_container = load_tomography_data(selected_file)
    if data_container is None:
        print("❌ Failed to load data. Exiting.")
        input("   Press Enter to exit...")
        return
    
    data_container['filename_npz'] = selected_file
    
    # Display dataset info
    cube = data_container['tomogram_cube']
    h, w, d = cube.shape
    print(f"   Dimensions: {h} pixels × {w} columns × {d} depth bins")
    print(f"   Data type: {cube.dtype}")
    print(f"   Data range: {np.min(cube):.3e} to {np.max(cube):.3e}")
    
    # Check z_vec - IMPORTANT: This is surface-relative!
    if 'z_vec' in data_container and data_container['z_vec'] is not None:
        z_vec = data_container['z_vec']
        print(f"\n   📊 SURFACE-RELATIVE VECTOR (z_vec) analysis:")
        print(f"      Shape: {z_vec.shape}")
        print(f"      Range: {z_vec[0]:.1f}m to {z_vec[-1]:.1f}m")
        print(f"      INTERPRETATION:")
        print(f"      • 0m = Ground surface")
        print(f"      • Negative values = Depth BELOW surface (e.g., {z_vec[0]:.0f}m)")
        print(f"      • Positive values = Height ABOVE surface (e.g., {z_vec[-1]:.0f}m)")
        print(f"      • Total range: {abs(z_vec[-1] - z_vec[0]):.0f}m")
        
        # Calculate statistics
        below_surface = z_vec[z_vec < 0]
        above_surface = z_vec[z_vec > 0]
        
        if len(below_surface) > 0:
            max_depth = abs(np.min(below_surface))
            print(f"      • Maximum depth: {max_depth:.0f}m below surface")
        
        if len(above_surface) > 0:
            max_height = np.max(above_surface)
            print(f"      • Maximum height: {max_height:.0f}m above surface")
    else:
        print(f"\n   ⚠️  No surface-relative vector (z_vec) found in data")
        print(f"      Defaulting to: -500m to +10m (510m total range)")
    
    # Downsampling option for large datasets with save option
    if h > 3000:
        print(f"\n⚠️  Large dataset detected ({h} pixels).")
        response = input("   Downsample for better performance? (y/n) [y]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            # Ask for downsampling factor
            while True:
                try:
                    factor_input = input("   Enter downsampling factor (e.g., 0.5 for half size, 0.25 for quarter): ").strip()
                    if factor_input == '':
                        scale = 2000 / h
                        print(f"   Using default factor: {scale:.3f}")
                        break
                    else:
                        scale = float(factor_input)
                        if 0.01 <= scale <= 1.0:
                            break
                        else:
                            print("   ❌ Please enter a value between 0.01 and 1.0")
                except ValueError:
                    print("   ❌ Please enter a valid number")
            
            print(f"   Downsampling by factor {scale:.3f}...")
            original_cube = cube.copy()
            data_container['tomogram_cube'] = zoom(cube, (scale, 1, 1), order=1)
            h_new = data_container['tomogram_cube'].shape[0]
            print(f"   New dimensions: {h_new} pixels × {w} columns × {d} depth bins")
            
            # Ask if user wants to save the downsampled file
            save_response = input("   Save downsampled version for later use? (y/n) [n]: ").strip().lower()
            if save_response in ['y', 'yes']:
                # Create filename for downsampled version
                base_name = os.path.splitext(selected_file)[0]
                downsampled_file = f"{base_name}_downsampled_{h_new}x{w}x{d}.npz"
                
                # Save all original data but with downsampled cube
                save_data = data_container.copy()
                save_data['tomogram_cube'] = original_cube  # Save original for reference
                save_data['downsampled_cube'] = data_container['tomogram_cube']  # Save downsampled
                save_data['downsampling_factor'] = scale
                save_data['original_dimensions'] = (h, w, d)
                
                try:
                    np.savez_compressed(downsampled_file, **save_data)
                    print(f"   ✅ Downsampled file saved as: {downsampled_file}")
                    print(f"   ⚠️  Note: Original cube preserved as 'tomogram_cube'")
                    print(f"   ⚠️  Note: Downsampled cube saved as 'downsampled_cube'")
                except Exception as e:
                    print(f"   ❌ Error saving downsampled file: {e}")
    
    # Geographic Integration
    print("\n🌍 Processing geographic coordinates...")
    
    # ========== FIX ADDED: Early coordinate validation and correction ==========
    geo_start = data_container.get('geo_start')
    geo_end = data_container.get('geo_end')
    
    if geo_start is not None and geo_end is not None:
        # Check for identical coordinates BEFORE any topography processing
        if geo_start[0] == geo_end[0] and geo_start[1] == geo_end[1]:
            print("   ⚠️  CRITICAL: Start and end coordinates are identical in NPZ file.")
            print("   ↪ Adding small offset (0.01°) to longitude for valid topography extraction...")
            # Create a new tuple with offset
            geo_end_corrected = (geo_end[0], geo_end[1] + 0.01)
            data_container['geo_end'] = geo_end_corrected
            print(f"   ↪ Corrected coordinates:")
            print(f"      Start: ({geo_start[0]:.6f}°, {geo_start[1]:.6f}°)")
            print(f"      End:   ({geo_end_corrected[0]:.6f}°, {geo_end_corrected[1]:.6f}°)")
        else:
            print(f"   📍 Start: ({geo_start[0]:.6f}°, {geo_start[1]:.6f}°)")
            print(f"   📍 End:   ({geo_end[0]:.6f}°, {geo_end[1]:.6f}°)")
    else:
        print("   ⚠️  No geographic coordinates found.")
        print("   Topography display will use flat elevation.")
        has_coords = False
    
    # Integrate coordinates (now with corrected coordinates if needed)
    data_container = integrate_topography_and_coordinates(data_container)
    
    # Re-fetch coordinates after potential correction
    geo_start = data_container.get('geo_start')
    geo_end = data_container.get('geo_end')
    has_coords = geo_start is not None and geo_end is not None
    
    if has_coords:
        # Calculate distance
        from PAR_capella_topography import calculate_section_length
        dist = calculate_section_length(geo_start, geo_end)
        print(f"   📏 Section length: {dist:.1f} meters")
        data_container['total_dist_m'] = dist
    
    # TOPOGRAPHY SELECTION
    print("\n" + "=" * 60)
    print("🌐 TOPOGRAPHY CONFIGURATION")
    print("=" * 60)
    
    # Get number of pixels for topography
    num_pixels = data_container['tomogram_cube'].shape[0]
    
    if not has_coords:
        print("   Using flat topography (no coordinates available)")
        data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
        data_container['flat_topo'] = True
        data_container['has_online_topo'] = False
        data_container['has_tiff_topo'] = False
        print(f"   ✅ Flat topography created: {num_pixels} points at 0.0m")
    else:
        print("\n   Select topography source:")
        print("   1. OpenTopoData API (Internet, free) - No exaggeration by default")
        print("   2. Local TIFF/DEM file (High precision) - No exaggeration by default")
        print("   3. Flat topography (0m elevation)")
        
        while True:
            choice = input("\n   Choice (1-3) [1]: ").strip()
            if choice == '':
                choice = '1'
                break
            elif choice in ['1', '2', '3']:
                break
            else:
                print("   ❌ Invalid. Enter 1, 2, or 3.")
        
        if choice == '1':
            print(f"\n🌐 Fetching elevation from OpenTopoData...")
            print("   Note: No vertical exaggeration applied by default")
            topo_profile = get_topography_from_opentopodata(geo_start, geo_end, num_pixels)
            
            if topo_profile is not None:
                data_container['online_elevation_profile'] = topo_profile
                data_container['has_online_topo'] = True
                data_container['has_tiff_topo'] = False
                data_container['flat_topo'] = False
                print(f"   ✅ Online topography loaded: {len(topo_profile)} points")
                print(f"   📊 Elevation: {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m")
                print(f"   📈 No exaggeration applied (1.0x)")
                
                # Store as the primary elevation profile
                data_container['elevation_profile'] = topo_profile
            else:
                print("   ⚠️  Failed to fetch online topography")
                print("   ↪ Using flat topography as fallback")
                data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                data_container['flat_topo'] = True
                data_container['has_online_topo'] = False
        
        elif choice == '2':
            if not HAS_RASTERIO:
                print("❌ rasterio not installed. Install with: pip install rasterio")
                print("   ↪ Falling back to OpenTopoData")
                topo_profile = get_topography_from_opentopodata(geo_start, geo_end, num_pixels)
                
                if topo_profile is not None:
                    data_container['online_elevation_profile'] = topo_profile
                    data_container['has_online_topo'] = True
                    data_container['has_tiff_topo'] = False
                    data_container['flat_topo'] = False
                    data_container['elevation_profile'] = topo_profile
                    print(f"   ✅ Online topography loaded (fallback)")
                    print(f"   📈 No exaggeration applied (1.0x)")
                else:
                    data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                    data_container['flat_topo'] = True
            else:
                # Look for TIFF files
                tiff_files = [f for f in os.listdir(cwd) if f.lower().endswith(('.tif', '.tiff', '.geotiff'))]
                
                if tiff_files:
                    print(f"\n📂 Found {len(tiff_files)} TIFF files:")
                    for i, f in enumerate(tiff_files):
                        print(f"   [{i+1}] {f}")
                    
                    try:
                        tiff_choice = input(f"\n   Select TIFF file (1-{len(tiff_files)}) or Enter to skip: ").strip()
                        if tiff_choice == '':
                            print("   Skipping TIFF, trying OpenTopoData...")
                            topo_profile = get_topography_from_opentopodata(geo_start, geo_end, num_pixels)
                            
                            if topo_profile is not None:
                                data_container['online_elevation_profile'] = topo_profile
                                data_container['has_online_topo'] = True
                                data_container['has_tiff_topo'] = False
                                data_container['flat_topo'] = False
                                data_container['elevation_profile'] = topo_profile
                                print(f"   📈 No exaggeration applied (1.0x)")
                            else:
                                data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                                data_container['flat_topo'] = True
                        else:
                            idx = int(tiff_choice) - 1
                            if 0 <= idx < len(tiff_files):
                                tiff_file = tiff_files[idx]
                                print(f"\n🌐 Loading elevation from {tiff_file}...")
                                print("   Note: No vertical exaggeration applied by default")
                                topo_profile = get_topography_from_tiff(
                                    os.path.join(cwd, tiff_file),
                                    geo_start,
                                    geo_end,
                                    num_pixels
                                )
                                
                                if topo_profile is not None:
                                    data_container['tiff_elevation_profile'] = topo_profile
                                    data_container['tiff_filename'] = tiff_file
                                    data_container['has_tiff_topo'] = True
                                    data_container['has_online_topo'] = False
                                    data_container['flat_topo'] = False
                                    data_container['elevation_profile'] = topo_profile
                                    print(f"   ✅ TIFF topography loaded: {len(topo_profile)} points")
                                    print(f"   📊 Elevation: {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m")
                                    print(f"   📈 No exaggeration applied (1.0x)")
                                else:
                                    print("   ⚠️  Failed to read TIFF")
                                    print("   ↪ Using flat topography")
                                    data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                                    data_container['flat_topo'] = True
                            else:
                                print("   ❌ Invalid selection")
                                print("   ↪ Using flat topography")
                                data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                                data_container['flat_topo'] = True
                    except ValueError:
                        print("   ❌ Invalid input")
                        print("   ↪ Using flat topography")
                        data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                        data_container['flat_topo'] = True
                else:
                    print("   ⚠️  No TIFF files found")
                    print("   ↪ Using flat topography")
                    data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
                    data_container['flat_topo'] = True
        
        elif choice == '3':
            print("   Using flat topography (0m elevation)")
            data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
            data_container['flat_topo'] = True
            data_container['has_online_topo'] = False
            data_container['has_tiff_topo'] = False
    
    # Ensure we have an elevation profile
    if 'elevation_profile' not in data_container:
        data_container['elevation_profile'] = create_flat_topography(num_pixels, 0.0)
        data_container['flat_topo'] = True
    
    # Verify topography profile
    topo_profile = data_container['elevation_profile']
    print(f"\n   📋 FINAL TOPOGRAPHY CONFIGURATION:")
    print(f"      Profile length: {len(topo_profile)} points")
    print(f"      Elevation range: {np.min(topo_profile):.1f}m to {np.max(topo_profile):.1f}m")
    print(f"      Source: {'Flat' if data_container.get('flat_topo', False) else 'Real'}")
    print(f"      Exaggeration: 1.0x (none applied by default)")
    
    # Display Settings
    print("\n" + "=" * 60)
    print("📊 DISPLAY SETTINGS")
    print("=" * 60)
    
    data_mag = np.abs(data_container['tomogram_cube'])
    p99 = np.percentile(data_mag, 99.5)
    
    print(f"   Data range (99.5th percentile): {p99:.3e}")
    
    use_log = input("   Use logarithmic scale? (y/n) [n]: ").strip().lower()
    log_mode = use_log in ['y', 'yes']
    
    # Set display preferences - default to no exaggeration
    data_container['display_prefs'] = {
        'log_mode': log_mode,
        'max_cutoff': float(p99),
        'min_cutoff': 0.0,
        'topo_exaggeration': 1.0  # CHANGED: Default to 1.0x (no exaggeration)
    }
    
    print(f"   Log scale: {'Yes' if log_mode else 'No'}")
    print(f"   Topography exaggeration: 1x (default - no exaggeration)")
    
    # Explain how tomogram will be displayed
    print(f"\n   📝 DISPLAY NOTES:")
    if 'z_vec' in data_container and data_container['z_vec'] is not None:
        z_vec = data_container['z_vec']
        print(f"      • Tomogram shows: {z_vec[0]:.0f}m to {z_vec[-1]:.0f}m relative to surface")
        print(f"      • 0m = Ground surface")
        print(f"      • Without topography: Shows surface-relative values")
        print(f"      • With topography: Converts to sea-level elevation")
        print(f"      • Field of view: Can be cropped in web interface")
        print(f"      • Topography exaggeration: OFF by default (1.0x)")
    
    # Start Server
    port = 5000
    url = f"http://127.0.0.1:{port}"
    
    print("\n" + "=" * 70)
    print("🚀 STARTING PAR-CAPELLA VIEWER")
    print("=" * 70)
    print(f"\n   Server URL: {url}")
    print("   Opening in browser...")
    print("   Press Ctrl+C to stop the server\n")
    
    # Open browser
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    # Start Flask server
    try:
        start_flask_server(data_container, port=port)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()