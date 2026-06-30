import os
import re
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix

def load_ompl(path, **kwargs):
    """
    Loads projection matrices from a text file with one matrix per line.
    Supports parsing metadata comments starting with '#>' or '#'.
    """
    Ps = []
    meta = {
        "spacing": 1.0,
        "detector_size_px": "400 300"
    }
    
    with open(path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith('#'):
                # Check for metadata attribute comments
                matches = re.findall(r'(\w+)="([^"]+)"', line_str)
                for k, v in matches:
                    meta[k] = v
                matches_unquoted = re.findall(r'(\w+)=([^"\s\[]+)', line_str)
                for k, v in matches_unquoted:
                    meta[k] = v
                matches_array = re.findall(r'(\w+)=\[([^\]]+)\]', line_str)
                for k, v in matches_array:
                    meta[k] = f"[{v}]"
                continue
                
            # Parse matrix row: e.g., [1 0 0 0; 0 1 0 0; 0 0 1 0] or just values
            clean = line_str.replace('[', '').replace(']', '').replace(';', ' ').strip()
            vals = np.array([float(x) for x in clean.split()])
            Ps.append(vals.reshape(3, 4))
   
    return [
        ProjectionMatrix(
            P,
            image_size=[int(x) for x in meta["detector_size_px"].split()],
            pixel_spacing=float(meta["spacing"])
        )
        for P in Ps
    ]

def save_ompl(Ps, path, first_line_comment="", spacing=0.0, detector_size_px=None, **kwargs):
    """
    Saves a list of ProjectionMatrix objects to a text file with one matrix per line.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        if first_line_comment:
            if not first_line_comment.startswith('#'):
                f.write(f"#{first_line_comment}\n")
            else:
                f.write(f"{first_line_comment}\n")
                
        # If spacing is provided, write metadata
        if spacing == 0.0 and Ps and isinstance(Ps[0], ProjectionMatrix):
            spacing = Ps[0].pixel_spacing
        if detector_size_px is None and Ps and isinstance(Ps[0], ProjectionMatrix):
            detector_size_px = Ps[0].image_size
            
        if spacing != 0.0 or detector_size_px is not None:
            meta_parts = []
            if spacing != 0.0:
                meta_parts.append(f'spacing="{spacing}"')
            if detector_size_px is not None:
                meta_parts.append(f'detector_size_px="{detector_size_px[0]} {detector_size_px[1]}"')
            f.write(f"#> {' '.join(meta_parts)}\n")
            
        for P in Ps:
            P_mat = P.P if isinstance(P, ProjectionMatrix) else P
            row0 = " ".join(f"{val:.12g}" for val in P_mat[0])
            row1 = " ".join(f"{val:.12g}" for val in P_mat[1])
            row2 = " ".join(f"{val:.12g}" for val in P_mat[2])
            f.write(f"[{row0}; {row1}; {row2}]\n")
    return True
