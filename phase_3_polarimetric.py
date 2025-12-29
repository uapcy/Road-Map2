# phase_3_polarimetric.py
# POLARIMETRIC TOMOGRAPHY MODULE
# Based on: 3-Multi-chromatic_analysis_polarimetric_interferometric_synthetic_aperture_radar...

import numpy as np
import warnings

def polarimetric_decomposition(Y_matrix_hh, Y_matrix_hv, Y_matrix_vv, method='pauli'):
    """
    Perform polarimetric decomposition for material classification.
    
    Args:
        Y_matrix_hh, Y_matrix_hv, Y_matrix_vv: Complex matrices for each polarization
        method: Decomposition method ('pauli', 'freeman', 'yamaguchi')
    
    Returns:
        dict: Decomposition components and scattering mechanisms
    """
    print("\n--- Polarimetric Decomposition ---")
    
    # Ensure all matrices have same shape
    shapes = [Y_matrix_hh.shape, Y_matrix_hv.shape, Y_matrix_vv.shape]
    if len(set(shapes)) > 1:
        raise ValueError("All polarization matrices must have the same shape")
    
    num_pixels, num_looks = Y_matrix_hh.shape
    
    if method.lower() == 'pauli':
        # Pauli decomposition
        k_p = (Y_matrix_hh + Y_matrix_vv) / np.sqrt(2)  # Single/odd bounce
        k_d = (Y_matrix_hh - Y_matrix_vv) / np.sqrt(2)  # Double/even bounce  
        k_t = np.sqrt(2) * Y_matrix_hv                  # Volume scattering
        
        decomposition = {
            'single_bounce': k_p,
            'double_bounce': k_d, 
            'volume_scattering': k_t,
            'method': 'pauli'
        }
        
        # Calculate scattering percentages
        power_p = np.mean(np.abs(k_p)**2, axis=1)
        power_d = np.mean(np.abs(k_d)**2, axis=1)
        power_t = np.mean(np.abs(k_t)**2, axis=1)
        total_power = power_p + power_d + power_t + 1e-9
        
        decomposition['percent_single'] = power_p / total_power
        decomposition['percent_double'] = power_d / total_power
        decomposition['percent_volume'] = power_t / total_power
        
        print(f"  Pauli decomposition complete")
        print(f"  Mean scattering percentages:")
        print(f"    Single bounce: {100*np.mean(decomposition['percent_single']):.1f}%")
        print(f"    Double bounce: {100*np.mean(decomposition['percent_double']):.1f}%")
        print(f"    Volume: {100*np.mean(decomposition['percent_volume']):.1f}%")
    
    elif method.lower() == 'freeman':
        # Freeman-Durden decomposition
        # Simplified implementation
        C_hh = np.mean(Y_matrix_hh * np.conj(Y_matrix_hh), axis=1)
        C_vv = np.mean(Y_matrix_vv * np.conj(Y_matrix_vv), axis=1)
        C_hv = np.mean(Y_matrix_hv * np.conj(Y_matrix_hv), axis=1)
        
        # Estimate scattering components
        volume = 2 * C_hv
        double_bounce = np.real(C_hh - volume/2)
        single_bounce = np.real(C_vv - volume/2)
        
        # Ensure non-negative
        double_bounce = np.maximum(double_bounce, 0)
        single_bounce = np.maximum(single_bounce, 0)
        
        total = volume + double_bounce + single_bounce + 1e-9
        
        decomposition = {
            'volume_scattering': volume,
            'double_bounce': double_bounce,
            'single_bounce': single_bounce,
            'percent_volume': volume / total,
            'percent_double': double_bounce / total,
            'percent_single': single_bounce / total,
            'method': 'freeman'
        }
    
    else:
        raise ValueError(f"Unsupported decomposition method: {method}")
    
    return decomposition

def polarimetric_tomography(Y_polarimetric_dict, sub_ap_centers, radar_params, **kwargs):
    """
    Perform polarimetric tomography using multi-polarization data.
    
    Args:
        Y_polarimetric_dict: Dictionary with polarization data
        sub_ap_centers: Sub-aperture centers
        radar_params: Radar parameters
        **kwargs: Tomography parameters
    
    Returns:
        dict: Polarimetric tomograms for each scattering mechanism
    """
    print("\n--- Polarimetric Tomography ---")
    
    # Get polarization data
    Y_hh = Y_polarimetric_dict.get('hh')
    Y_hv = Y_polarimetric_dict.get('hv')
    Y_vv = Y_polarimetric_dict.get('vv')
    
    if Y_hh is None:
        raise ValueError("HH polarization data required")
    
    # Get tomography parameters
    from phase_3_base_tomography import focus_sonic_tomogram_enhanced
    
    # Process each polarization
    tomograms = {}
    
    for pol_name, Y_pol in [('hh', Y_hh), ('hv', Y_hv), ('vv', Y_vv)]:
        if Y_pol is None:
            continue
            
        print(f"  Processing {pol_name.upper()} polarization...")
        
        # Run tomography for this polarization
        tomogram, z_vec, _ = focus_sonic_tomogram_enhanced(
            Y_pol, sub_ap_centers, radar_params, **kwargs
        )
        
        tomograms[f'tomogram_{pol_name}'] = tomogram
        tomograms[f'z_vec_{pol_name}'] = z_vec
    
    # Store common z_vec if available
    if 'z_vec_hh' in tomograms:
        tomograms['z_vec'] = tomograms['z_vec_hh']
    
    print(f"  Polarimetric tomography complete for {len(tomograms)//2} polarizations")
    
    return tomograms

def classify_materials(polarimetric_features, method='wishart'):
    """
    Classify materials based on polarimetric features.
    
    Args:
        polarimetric_features: Dictionary of polarimetric features
        method: Classification method ('wishart', 'svm', 'kmeans')
    
    Returns:
        np.ndarray: Material classification labels
        dict: Classification probabilities
    """
    print("\n--- Material Classification ---")
    
    # Extract features
    if 'percent_single' in polarimetric_features:
        features = np.column_stack([
            polarimetric_features['percent_single'],
            polarimetric_features['percent_double'],
            polarimetric_features['percent_volume']
        ])
    else:
        # Use raw scattering components
        features = np.column_stack([
            np.mean(np.abs(polarimetric_features.get('single_bounce', 0)), axis=1),
            np.mean(np.abs(polarimetric_features.get('double_bounce', 0)), axis=1),
            np.mean(np.abs(polarimetric_features.get('volume_scattering', 0)), axis=1)
        ])
    
    num_pixels = features.shape[0]
    
    if method.lower() == 'wishart':
        # Simplified Wishart-like classification
        # Based on scattering mechanism dominance
        
        labels = np.zeros(num_pixels, dtype=int)
        probabilities = np.zeros((num_pixels, 4))  # 4 classes
        
        for i in range(num_pixels):
            p_single = features[i, 0]
            p_double = features[i, 1]
            p_volume = features[i, 2]
            
            # Simple rule-based classification
            if p_volume > 0.7:
                labels[i] = 0  # Vegetation/volume scattering
                probabilities[i, 0] = 1.0
            elif p_double > 0.6:
                labels[i] = 1  # Urban/double bounce
                probabilities[i, 1] = 1.0
            elif p_single > 0.6:
                labels[i] = 2  # Surface/single bounce
                probabilities[i, 2] = 1.0
            else:
                labels[i] = 3  # Mixed/unknown
                probabilities[i, 3] = 1.0
        
        class_names = ['Vegetation', 'Urban', 'Surface', 'Mixed']
        
    else:
        raise ValueError(f"Unsupported classification method: {method}")
    
    # Count classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Material classification complete:")
    for label, count in zip(unique_labels, counts):
        class_name = ['Vegetation', 'Urban', 'Surface', 'Mixed'][label]
        percentage = 100 * count / num_pixels
        print(f"    {class_name}: {count} pixels ({percentage:.1f}%)")
    
    return labels, {'probabilities': probabilities, 'class_names': class_names}