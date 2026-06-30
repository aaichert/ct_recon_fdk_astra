import os
import re
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix

def load_conrad(path, spacing=1.0, detector_size_px=(400, 300), **kwargs):
    """
    Loads projection matrices from a Stanford CONRAD XML configuration file.
    """
    if 'pixel_spacing' in kwargs:
        spacing = kwargs['pixel_spacing']
    if 'detector_size_px' in kwargs:
        detector_size_px = kwargs['detector_size_px']
        
    Ps = []
    with open(path, 'r') as f:
        content = f.read()
        
    # Find all edu.stanford.rsl.conrad.geometry.Projection object blocks
    # Specifically, find the string values inside PMatrixSerialization property voids
    pattern = r'property="PMatrixSerialization"\s*>\s*<string>\s*(.*?)\s*</string>'
    matrices_str = re.findall(pattern, content, re.DOTALL)
    
    for m_str in matrices_str:
        clean = m_str.replace('[', '').replace(']', '').replace(';', ' ').strip()
        if not clean:
            continue
        vals = [float(x) for x in clean.split()]
        if len(vals) == 12:
            P_mat = np.array(vals).reshape(3, 4)
            Ps.append(ProjectionMatrix(P_mat, image_size=detector_size_px, pixel_spacing=spacing))
            
    return Ps

def save_conrad(Ps, path, **kwargs):
    """
    Saves a list of ProjectionMatrix objects to a Stanford CONRAD XML file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    xml_lines = []
    xml_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml_lines.append('<java version="1.8.0_60" class="java.beans.XMLDecoder">')
    xml_lines.append(f' <array class="edu.stanford.rsl.conrad.geometry.Projection" length="{len(Ps)}">')
    
    for i, P in enumerate(Ps):
        P_mat = P.P if isinstance(P, ProjectionMatrix) else P
        row0 = " ".join(f"{val:.12g}" for val in P_mat[0])
        row1 = " ".join(f"{val:.12g}" for val in P_mat[1])
        row2 = " ".join(f"{val:.12g}" for val in P_mat[2])
        matrix_str = f"[[{row0}]; [{row1}]; [{row2}]]"
        
        xml_lines.append(f'  <void index="{i}">')
        xml_lines.append('   <object class="edu.stanford.rsl.conrad.geometry.Projection">')
        xml_lines.append('    <void property="PMatrixSerialization">')
        xml_lines.append(f'     <string>{matrix_str}</string>')
        xml_lines.append('    </void>')
        xml_lines.append('   </object>')
        xml_lines.append('  </void>')
        
    xml_lines.append(' </array>')
    xml_lines.append('</java>')
    
    with open(path, 'w') as f:
        f.write("\n".join(xml_lines) + "\n")
    return True
