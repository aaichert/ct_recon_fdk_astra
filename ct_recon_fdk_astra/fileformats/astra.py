import os
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix

def load_astra(path, image_size=(400, 300), pixel_spacing=1.0, swap_y_z=True, **kwargs):
    """
    Loads projection matrices from an ASTRA vector geometry file (.txt or .vec).
    Each line in the file is expected to contain 12 floating point values:
    S_x S_y S_z D_x D_y D_z u_x u_y u_z v_x v_y v_z
    """
    if 'detector_size_px' in kwargs:
        image_size = kwargs['detector_size_px']
    if 'pixel_spacing' in kwargs:
        pixel_spacing = kwargs['pixel_spacing']
    elif 'spacing' in kwargs:
        pixel_spacing = kwargs['spacing']
        
    Ps = []
    W, H = image_size
    
    with open(path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
            vals = [float(x) for x in line_str.replace(',', ' ').split()]
            if len(vals) == 12:
                S = np.array(vals[0:3])
                D = np.array(vals[3:6])
                u = np.array(vals[6:9])
                v = np.array(vals[9:12])
                
                if swap_y_z:
                    # swap Y and Z back
                    S = np.array([S[0], S[2], S[1]])
                    D = np.array([D[0], D[2], D[1]])
                    u = np.array([u[0], u[2], u[1]])
                    v = np.array([v[0], v[2], v[1]])
                    
                w = np.cross(u, v)
                R_eff = np.vstack([u / np.dot(u, u), v / np.dot(v, v), w / np.dot(w, w)])
                d = R_eff @ (D - S)
                d_u, d_v, d_w = d[0], d[1], d[2]
                u_mid = (W - 1) / 2.0
                v_mid = (H - 1) / 2.0
                K_eff = np.array([
                    [d_w, 0.0, u_mid - d_u],
                    [0.0, d_w, v_mid - d_v],
                    [0.0, 0.0, 1.0]
                ])
                P_mat = K_eff @ R_eff @ np.hstack([np.eye(3), -S.reshape(3, 1)])
                Ps.append(ProjectionMatrix(P_mat, image_size=image_size, pixel_spacing=pixel_spacing))
    return Ps

def save_astra(Ps, path, swap_y_z=True, **kwargs):
    """
    Saves a list of ProjectionMatrix objects to an ASTRA vector geometry file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for P in Ps:
            P_mat = P.P if isinstance(P, ProjectionMatrix) else P
            W, H = P.image_size if isinstance(P, ProjectionMatrix) else (400, 300)
            
            P_obj = P if isinstance(P, ProjectionMatrix) else ProjectionMatrix(P_mat, image_size=(W, H), pixel_spacing=1.0)
            K, R, t, _ = P_obj.decomposition(imageVPointsUp=False)
            
            C_hom = P_obj.getCenterOfProjection().flatten()
            S = C_hom[:3] / (C_hom[3] if abs(C_hom[3]) > 1e-12 else 1.0)
            
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            
            r1 = R[0, :]
            r2 = R[1, :]
            r3 = R[2, :]
            
            d_dist = 1.0
            D_pp = S + d_dist * r3
            u = (d_dist / fx) * r1
            v = (d_dist / fy) * r2
            
            u_mid = (W - 1) / 2.0
            v_mid = (H - 1) / 2.0
            D = D_pp + (u_mid - cx) * u + (v_mid - cy) * v
            
            if swap_y_z:
                S = np.array([S[0], S[2], S[1]])
                D = np.array([D[0], D[2], D[1]])
                u = np.array([u[0], u[2], u[1]])
                v = np.array([v[0], v[2], v[1]])
                
            vals = np.concatenate([S, D, u, v])
            row_str = " ".join(f"{val:.12g}" for val in vals)
            f.write(row_str + "\n")
    return True
