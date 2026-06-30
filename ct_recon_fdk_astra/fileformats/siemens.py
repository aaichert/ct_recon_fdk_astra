import os
import re
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from .utils import ctCircularTrajectoryToParameters

def load_projtable_xml(path, spacing=1.0, detector_size_px=(400, 300), **kwargs):
    """
    Loads projection matrices from a Siemens projtable.xml file (v1.3 style).
    """
    if 'pixel_spacing' in kwargs:
        spacing = kwargs['pixel_spacing']
    if 'detector_size_px' in kwargs:
        detector_size_px = kwargs['detector_size_px']
        
    Ps = []
    with open(path, 'r') as f:
        content = f.read()
        
    blocks = re.findall(r'<ProjectionMatrix>\s*(.*?)\s*</ProjectionMatrix>', content, re.DOTALL)
    for b in blocks:
        vals = [float(x) for x in b.replace('\n', ' ').split()]
        if len(vals) == 12:
            P_mat = np.array(vals).reshape(3, 4)
            Ps.append(ProjectionMatrix(P_mat, image_size=detector_size_px, pixel_spacing=spacing))
    return Ps

def load_projtable_txt(path, spacing=1.0, detector_size_px=(400, 300), **kwargs):
    """
    Loads projection matrices from a Siemens projtable.txt file.
    """
    if 'pixel_spacing' in kwargs:
        spacing = kwargs['pixel_spacing']
    if 'detector_size_px' in kwargs:
        detector_size_px = kwargs['detector_size_px']
        
    Ps = []
    with open(path, 'r') as f:
        lines = f.readlines()
        
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
            
        if line.startswith('@'):
            # The next line (idx+1) contains the primary and secondary angles.
            # The lines after that (idx+2, idx+3, idx+4) contain the matrix rows.
            if idx + 4 < len(lines):
                row0 = [float(x) for x in lines[idx+2].strip().split()]
                row1 = [float(x) for x in lines[idx+3].strip().split()]
                row2 = [float(x) for x in lines[idx+4].strip().split()]
                P_mat = np.vstack([row0, row1, row2])
                Ps.append(ProjectionMatrix(P_mat, image_size=detector_size_px, pixel_spacing=spacing))
            idx += 5
        else:
            idx += 1
    return Ps

def save_projtable_txt(Ps, path, spacing=1.0, **kwargs):
    """
    Saves a list of ProjectionMatrix objects to a Siemens projtable.txt file (version 3 format).
    """
    if 'pixel_spacing' in kwargs:
        spacing = kwargs['pixel_spacing']
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Compute primary and secondary angles from trajectory
    params = ctCircularTrajectoryToParameters(Ps, detector_pixel_spacing=spacing)
    primary_angles = params.get("primary_angles", [0.0] * len(Ps))
    secondary_angles = params.get("secondary_angles", [0.0] * len(Ps))
    
    with open(path, 'w') as file:
        file.write("projtable version 3\n")
        file.write("Mon Jan 01 1000 ompl2projtable\n \n")
        file.write("# format: angle/entries of projection matrices\n")
        file.write(f"{len(Ps)}\n\n")
        
        for i, P in enumerate(Ps):
            P_mat = P.P if isinstance(P, ProjectionMatrix) else P
            norm_val = np.linalg.norm(P_mat[2, :3])
            P_norm = P_mat / (norm_val if norm_val > 1e-8 else 1.0)
            if np.abs(P_norm[2, 3]) > 1e-8:
                P_norm = P_norm / P_norm[2, 3]
                
            file.write(f"@ {i}\n")
            file.write(f"{primary_angles[i]:.6f} {secondary_angles[i]:.6f} \n")
            file.write(f"{P_norm[0,0]:.12g} {P_norm[0,1]:.12g} {P_norm[0,2]:.12g} {P_norm[0,3]:.12g}\n")
            file.write(f"{P_norm[1,0]:.12g} {P_norm[1,1]:.12g} {P_norm[1,2]:.12g} {P_norm[1,3]:.12g}\n")
            file.write(f"{P_norm[2,0]:.12g} {P_norm[2,1]:.12g} {P_norm[2,2]:.12g} {P_norm[2,3]:.12g}\n\n")
    return True

def save_projtable_xml(Ps, path, **kwargs):
    """
    Saves a list of ProjectionMatrix objects to a Siemens projtable.xml file (v1.3 style).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xml_lines = []
    xml_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml_lines.append('<Projtable>')
    for P in Ps:
        P_mat = P.P if isinstance(P, ProjectionMatrix) else P
        xml_lines.append('  <ProjectionMatrix>')
        row0 = " ".join(f"{val:.12g}" for val in P_mat[0])
        row1 = " ".join(f"{val:.12g}" for val in P_mat[1])
        row2 = " ".join(f"{val:.12g}" for val in P_mat[2])
        xml_lines.append(f'    {row0}')
        xml_lines.append(f'    {row1}')
        xml_lines.append(f'    {row2}')
        xml_lines.append('  </ProjectionMatrix>')
    xml_lines.append('</Projtable>')
    
    with open(path, 'w') as f:
        f.write("\n".join(xml_lines) + "\n")
    return True
