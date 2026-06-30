import os
import re
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from .utils import ctCircularTrajectoryToParameters

def load_rtk(path, spacing=1.0, detector_size_px=(400, 300), **kwargs):
    """
    Loads projection matrices from an RTK geometry XML file.
    """
    if 'pixel_spacing' in kwargs:
        spacing = kwargs['pixel_spacing']
    if 'detector_size_px' in kwargs:
        detector_size_px = kwargs['detector_size_px']
        
    Ps = []
    with open(path, 'r') as f:
        content = f.read()
        
    blocks = re.findall(r'<Matrix>\s*(.*?)\s*</Matrix>', content, re.DOTALL)
    for b in blocks:
        vals = [float(x) for x in b.replace('\n', ' ').split()]
        if len(vals) == 12:
            P_mat = np.array(vals).reshape(3, 4)
            Ps.append(ProjectionMatrix(P_mat, image_size=detector_size_px, pixel_spacing=spacing))
    return Ps

def save_rtk(Ps, path, spacing=1.0, **kwargs):
    """
    Saves a list of ProjectionMatrix objects to an RTK geometry XML file.
    """
    if 'pixel_spacing' in kwargs:
        spacing = kwargs['pixel_spacing']
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    params = ctCircularTrajectoryToParameters(Ps, detector_pixel_spacing=spacing)
    sids = params.get("source_to_iso_center_distances", [1000.0] * len(Ps))
    sdds = params.get("source_to_detector_distances", [1536.0] * len(Ps))
    primary_angles = params.get("primary_angles", [0.0] * len(Ps))
    secondary_angles = params.get("secondary_angles", [0.0] * len(Ps))
    
    global_sid = np.mean(sids) if sids else 1000.0
    global_sdd = np.mean(sdds) if sdds else 1536.0
    
    xml_lines = []
    xml_lines.append('<?xml version="1.0"?>')
    xml_lines.append('<!DOCTYPE RTKGEOMETRY>')
    xml_lines.append('<RTKThreeDCircularGeometry version="3">')
    xml_lines.append(f'  <SourceToIsocenterDistance>{global_sid:.6f}</SourceToIsocenterDistance>')
    xml_lines.append(f'  <SourceToDetectorDistance>{global_sdd:.6f}</SourceToDetectorDistance>')
    
    for i, P in enumerate(Ps):
        P_mat = P.P if isinstance(P, ProjectionMatrix) else P
        
        row0 = " ".join(f"{val:.12g}" for val in P_mat[0])
        row1 = " ".join(f"{val:.12g}" for val in P_mat[1])
        row2 = " ".join(f"{val:.12g}" for val in P_mat[2])
        matrix_str = f"{row0}\n      {row1}\n      {row2}"
        
        gantry = primary_angles[i]
        out_of_plane = secondary_angles[i]
        
        xml_lines.append('  <Projection>')
        xml_lines.append(f'    <GantryAngle>{gantry:.12g}</GantryAngle>')
        xml_lines.append(f'    <OutOfPlaneAngle>{out_of_plane:.12g}</OutOfPlaneAngle>')
        xml_lines.append('    <InPlaneAngle>0</InPlaneAngle>')
        xml_lines.append('    <SourceOffsetX>0</SourceOffsetX>')
        xml_lines.append('    <SourceOffsetY>0</SourceOffsetY>')
        xml_lines.append('    <ProjectionOffsetX>0</ProjectionOffsetX>')
        xml_lines.append('    <ProjectionOffsetY>0</ProjectionOffsetY>')
        xml_lines.append('    <Matrix>')
        xml_lines.append(f'      {matrix_str}')
        xml_lines.append('    </Matrix>')
        xml_lines.append('  </Projection>')
        
    xml_lines.append('</RTKThreeDCircularGeometry>')
    
    with open(path, 'w') as f:
        f.write("\n".join(xml_lines) + "\n")
    return True
