# PAR_capella_filters.py
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

def apply_processing_pipeline(cube, params, aux_data=None):
    """
    Apply filters to the 3D data cube based on dictionary parameters.
    Includes both original Capella filters and ported AN filters.
    """
    processed = cube.copy()
    
    # --- 1. AN: Multi-Notch Vertical Stripe Filter ---
    # Replaces simple SVD if 'enable_vertical' is true
    if params.get('enable_vertical', False):
        num_notches = int(params.get('notch_count', 3))
        notch_width = int(params.get('notch_width', 2))
        
        # Apply to each depth slice (or column-wise on the whole cube?)
        # Vertical stripes appear in the Cross-Track (X) vs Depth (Z) or Along-Track (Y) vs Depth (Z).
        # In AN_filters, it processed row-by-row.
        # Here we process the cube. We iterate through the 'Along-Track' (Y) axis 
        # to fix stripes in the 'Cross-Track' (X) slices.
        
        rows, cols, depths = processed.shape
        # Process each "Along-Track" slice (Vertical slice X-Z plane)
        for c in range(cols):
            slice_data = processed[:, c, :].T # Transpose to get (Depth, Azimuth) -> (Rows, Cols) for AN logic
            # AN logic iterates rows (Depth levels) and FFTs the columns (Azimuth)?
            # Wait, vertical stripes usually mean constant noise at specific azimuths.
            # In V7 cube: Axis 0=Pixels(Azimuth?), Axis 1=Cols(Range?), Axis 2=Depth?
            # We assume standard orientation: (Azimuth, Range, Depth).
            
            # Let's apply the AN logic to every "Depth" row of the Azimuth-Range plane
            # To match AN's behavior, we apply it to the data view being requested.
            # Ideally, we filter the whole cube.
            pass 
            
        # Simplified implementation of AN's Multi-Notch for the whole cube
        # We assume stripes are constant along Axis 0 (Azimuth/Pixels).
        # We take the mean along Axis 0 to find static frequencies?
        # Actually, let's stick to the AN logic: Process row independently.
        
        # We will process each Depth Slice (Axis 2)
        for d in range(depths):
            img = processed[:, :, d] # (Rows, Cols)
            # AN Logic: "Process each row independently".
            # If stripes are vertical, we need to filter the *rows* of the image (horizontal freq).
            # If stripes are horizontal, filter columns.
            # Assuming 'Vertical Stripe' means noise constant in Y, varying in X.
            
            # Apply FFT row-by-row (Axis 0 of the image)
            for r in range(img.shape[0]):
                row_data = img[r, :]
                fft = np.fft.fft(row_data)
                mag = np.abs(fft)
                
                # Find peaks (DC is 0)
                # Zero out 'num_notches' highest peaks excluding DC
                peaks = np.argsort(mag[1:len(mag)//2])[-num_notches:] + 1
                
                for p in peaks:
                    # Notch width
                    p_start = max(1, p - notch_width)
                    p_end = min(len(fft)//2, p + notch_width + 1)
                    fft[p_start:p_end] = 0
                    fft[-p_end:-p_start] = 0
                
                img[r, :] = np.real(np.fft.ifft(fft))
            
            processed[:, :, d] = img

    # --- 2. Capella: Gaussian Denoise ---
    sigma = float(params.get('denoise_sigma', 0.0))
    if sigma > 0:
        processed = gaussian_filter(processed, sigma=sigma)

    # --- 3. AN: Depth-Band Filter ---
    # Zero out data outside specific Z-index range
    if params.get('enable_depth_band', False):
        d_min = int(params.get('depth_min_idx', 0))
        d_max = int(params.get('depth_max_idx', processed.shape[2]))
        
        # Zero out above min
        if d_min > 0:
            processed[:, :, :d_min] = 0
        # Zero out below max
        if d_max < processed.shape[2]:
            processed[:, :, d_max:] = 0

    # --- 4. AN: Coherence/Robustness Masking ---
    if params.get('enable_masking', False) and aux_data:
        mask_type = params.get('mask_type', 'Coherence') # or 'Robustness'
        threshold = float(params.get('mask_threshold', 0.5))
        
        if mask_type == 'Coherence' and 'coherence_cube' in aux_data:
            mask_vol = aux_data['coherence_cube']
            mask = mask_vol < threshold
            processed[mask] = 0
        elif mask_type == 'Robustness' and 'robustness_score_3d' in aux_data:
            mask_vol = aux_data['robustness_score_3d']
            mask = mask_vol < threshold
            processed[mask] = 0

    return processed