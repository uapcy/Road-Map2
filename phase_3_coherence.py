# phase_3_coherence.py
# ENHANCED WITH SQUEE SAR PHASE LINKING ALGORITHM
# Based on: N-A_New_Algorithm_for_Processing_Interferometric_Data-Stacks_SqueeSAR.pdf

import numpy as np

def calculate_coherence_map(Y_matrix):
    """
    Calculates the infra-chromatic coherence for each pixel row in the Y matrix.

    Coherence is defined as the magnitude of the complex sum divided by the
    sum of the magnitudes. It is a measure of the phase stability of the signal.
    A value of 1.0 indicates a perfectly stable phase, while 0.0 indicates
    a completely random phase.

    Args:
        Y_matrix (np.ndarray): The complex-valued displacement matrix, with
                               shape (num_pixels, num_looks).

    Returns:
        np.ndarray: A 1D array of coherence values, with shape (num_pixels,).
    """
    print("\n--- Calculating Infra-Chromatic Coherence Map ---", flush=True)

    # Calculate the magnitude of the complex sum of the vectors (numerator)
    # This is |y1 + y2 + ... + yn|
    numerator = np.abs(np.sum(Y_matrix, axis=1))

    # Calculate the sum of the magnitudes of the vectors (denominator)
    # This is |y1| + |y2| + ... + |yn|
    denominator = np.sum(np.abs(Y_matrix), axis=1)

    # Initialize coherence map
    num_pixels = Y_matrix.shape[0]
    coherence_map = np.zeros(num_pixels, dtype=np.float32)

    # Avoid division by zero for pixels with no signal
    valid_indices = denominator > 1e-9
    coherence_map[valid_indices] = numerator[valid_indices] / denominator[valid_indices]

    print("--- Coherence Map Calculation Complete ---", flush=True)
    return coherence_map

def calculate_amplitude_dispersion_index(Y_matrix):
    """
    Calculate Amplitude Dispersion Index (ADI) for each pixel.
    Based on SqueeSAR algorithm: PS selection using amplitude stability.
    
    ADI = σ_A / μ_A, where lower ADI indicates better Persistent Scatterer (PS)
    
    Args:
        Y_matrix (np.ndarray): Complex-valued displacement matrix [num_pixels, num_looks]
    
    Returns:
        np.ndarray: ADI values for each pixel [num_pixels]
    """
    print("\n--- Calculating Amplitude Dispersion Index (SqueeSAR PS Selection) ---")
    
    # Calculate amplitude for each look
    amplitudes = np.abs(Y_matrix)
    
    # Calculate mean and standard deviation of amplitudes
    mean_amplitude = np.mean(amplitudes, axis=1)
    std_amplitude = np.std(amplitudes, axis=1)
    
    # Calculate ADI (avoid division by zero)
    adi = std_amplitude / (mean_amplitude + 1e-9)
    
    print(f"  ADI range: {np.min(adi):.3f} to {np.max(adi):.3f}")
    print(f"  Mean ADI: {np.mean(adi):.3f}")
    
    return adi

def identify_persistent_scatterers(Y_matrix, adi_threshold=0.25):
    """
    Identify Persistent Scatterers (PS) using Amplitude Dispersion Index.
    Based on SqueeSAR algorithm for PS selection.
    
    Args:
        Y_matrix (np.ndarray): Complex-valued displacement matrix
        adi_threshold (float): Threshold for PS selection (lower = stricter)
    
    Returns:
        np.ndarray: Boolean mask of persistent scatterers [num_pixels]
        np.ndarray: ADI values for each pixel
    """
    adi = calculate_amplitude_dispersion_index(Y_matrix)
    
    # PS selection: ADI < threshold
    ps_mask = adi < adi_threshold
    
    num_ps = np.sum(ps_mask)
    total_pixels = len(ps_mask)
    
    print(f"\n--- Persistent Scatterer Identification ---")
    print(f"  Total pixels: {total_pixels}")
    print(f"  PS identified: {num_ps} ({100*num_ps/total_pixels:.1f}%)")
    print(f"  ADI threshold: {adi_threshold}")
    
    return ps_mask, adi

def estimate_phase_linking_matrix(Y_matrix, ps_mask=None, max_iterations=20, convergence_threshold=1e-6):
    """
    Estimate optimal phase linking matrix using SqueeSAR algorithm.
    This improves coherence estimation by considering temporal decorrelation.
    
    Based on: N-A_New_Algorithm_for_Processing_Interferometric_Data-Stacks_SqueeSAR.pdf
    
    Args:
        Y_matrix (np.ndarray): Complex-valued displacement matrix [num_pixels, num_looks]
        ps_mask (np.ndarray): Boolean mask of persistent scatterers (optional)
        max_iterations (int): Maximum iterations for convergence
        convergence_threshold (float): Convergence criterion
    
    Returns:
        np.ndarray: Phase linking matrix [num_looks, num_looks]
        np.ndarray: Estimated deformation phases [num_pixels, num_looks]
    """
    print("\n--- Estimating Phase Linking Matrix (SqueeSAR Algorithm) ---")
    
    num_pixels, num_looks = Y_matrix.shape
    
    # If PS mask not provided, use all pixels (but weight by amplitude stability)
    if ps_mask is None:
        ps_mask = np.ones(num_pixels, dtype=bool)
    
    # Use only PS pixels for phase linking estimation
    Y_ps = Y_matrix[ps_mask, :]
    num_ps = Y_ps.shape[0]
    
    if num_ps == 0:
        print("  WARNING: No persistent scatterers found. Using all pixels.")
        Y_ps = Y_matrix
        num_ps = num_pixels
    
    print(f"  Using {num_ps} pixels for phase linking estimation")
    
    # Initialize phase linking matrix as identity
    Phi = np.eye(num_looks, dtype=np.complex64)
    
    # Initialize deformation phases
    deformation_phases = np.ones((num_pixels, num_looks), dtype=np.complex64)
    
    # Calculate amplitude weights (more stable amplitudes get higher weight)
    amplitudes = np.abs(Y_ps)
    amplitude_stability = 1.0 / (np.std(amplitudes, axis=1) / np.mean(amplitudes, axis=1) + 1e-9)
    weights = amplitude_stability / np.sum(amplitude_stability)
    
    # Iterative optimization
    for iteration in range(max_iterations):
        # Step 1: Estimate deformation phases given current Phi
        for i in range(num_ps):
            # Weighted phase estimation
            y_i = Y_ps[i, :]
            deformation_phases[i, :] = np.exp(1j * np.angle(Phi @ y_i))
        
        # Step 2: Update Phi given estimated deformation phases
        Phi_old = Phi.copy()
        
        # Calculate correlation matrix
        R = np.zeros((num_looks, num_looks), dtype=np.complex64)
        for i in range(num_ps):
            y_i = Y_ps[i, :]
            d_i = deformation_phases[i, :]
            R += weights[i] * np.outer(y_i, np.conj(y_i)) / np.outer(d_i, np.conj(d_i))
        
        # Normalize
        R = R / np.trace(R) * num_looks
        
        # Update Phi using eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Use dominant eigenvector for phase linking
        dominant_idx = np.argmax(np.abs(eigenvalues))
        phi_vector = eigenvectors[:, dominant_idx]
        
        # Update phase linking matrix
        Phi = np.diag(phi_vector / phi_vector[0])  # Normalize to first element
        
        # Check convergence
        delta = np.linalg.norm(Phi - Phi_old) / np.linalg.norm(Phi_old)
        
        if delta < convergence_threshold:
            print(f"  Convergence reached at iteration {iteration+1} (Δ={delta:.2e})")
            break
    
    print(f"  Phase linking matrix estimated ({iteration+1} iterations)")
    
    return Phi, deformation_phases

def apply_phase_linking(Y_matrix, Phi=None, ps_mask=None):
    """
    Apply phase linking to improve coherence and phase estimation.
    
    Args:
        Y_matrix (np.ndarray): Complex-valued displacement matrix
        Phi (np.ndarray): Phase linking matrix (optional, will estimate if None)
        ps_mask (np.ndarray): Persistent scatterer mask (optional)
    
    Returns:
        np.ndarray: Phase-linked displacement matrix
        np.ndarray: Improved coherence map
        dict: Diagnostic information
    """
    print("\n--- Applying Phase Linking (SqueeSAR Enhancement) ---")
    
    # Estimate phase linking matrix if not provided
    if Phi is None:
        Phi, deformation_phases = estimate_phase_linking_matrix(Y_matrix, ps_mask)
    else:
        # Estimate deformation phases using given Phi
        num_pixels, num_looks = Y_matrix.shape
        deformation_phases = np.ones((num_pixels, num_looks), dtype=np.complex64)
        for i in range(num_pixels):
            y_i = Y_matrix[i, :]
            deformation_phases[i, :] = np.exp(1j * np.angle(Phi @ y_i))
    
    # Apply phase correction
    Y_linked = Y_matrix.copy()
    for i in range(Y_matrix.shape[0]):
        Y_linked[i, :] = Y_matrix[i, :] * np.conj(deformation_phases[i, :])
    
    # Calculate improved coherence
    coherence_improved = calculate_coherence_map(Y_linked)
    
    # Calculate coherence improvement
    coherence_original = calculate_coherence_map(Y_matrix)
    coherence_gain = coherence_improved - coherence_original
    
    diagnostics = {
        'coherence_original': coherence_original,
        'coherence_improved': coherence_improved,
        'coherence_gain': coherence_gain,
        'mean_coherence_original': np.mean(coherence_original),
        'mean_coherence_improved': np.mean(coherence_improved),
        'mean_coherence_gain': np.mean(coherence_gain),
        'phase_linking_matrix': Phi,
        'deformation_phases': deformation_phases
    }
    
    print(f"  Mean coherence: {np.mean(coherence_original):.3f} -> {np.mean(coherence_improved):.3f}")
    print(f"  Coherence gain: {np.mean(coherence_gain):.3f}")
    
    return Y_linked, coherence_improved, diagnostics

def calculate_temporal_coherence(Y_matrix, window_size=5):
    """
    Calculate temporal coherence (coherence over time).
    Useful for identifying stable scatterers and deformation patterns.
    
    Args:
        Y_matrix (np.ndarray): Complex-valued displacement matrix
        window_size (int): Window size for temporal averaging
    
    Returns:
        np.ndarray: Temporal coherence for each pixel [num_pixels]
        np.ndarray: Temporal coherence matrix [num_pixels, num_looks-window_size+1]
    """
    print("\n--- Calculating Temporal Coherence ---")
    
    num_pixels, num_looks = Y_matrix.shape
    
    if num_looks < window_size:
        print(f"  WARNING: Not enough looks ({num_looks}) for window size {window_size}")
        return np.ones(num_pixels), np.ones((num_pixels, 1))
    
    # Calculate temporal coherence using sliding window
    num_windows = num_looks - window_size + 1
    temporal_coherence_matrix = np.zeros((num_pixels, num_windows))
    
    for w in range(num_windows):
        window_data = Y_matrix[:, w:w+window_size]
        window_coherence = calculate_coherence_map(window_data)
        temporal_coherence_matrix[:, w] = window_coherence
    
    # Average temporal coherence for each pixel
    temporal_coherence = np.mean(temporal_coherence_matrix, axis=1)
    
    print(f"  Mean temporal coherence: {np.mean(temporal_coherence):.3f}")
    print(f"  Temporal coherence range: [{np.min(temporal_coherence):.3f}, {np.max(temporal_coherence):.3f}]")
    
    return temporal_coherence, temporal_coherence_matrix