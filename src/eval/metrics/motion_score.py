import numpy as np
from dataclasses import dataclass

@dataclass
class MotionFeature:
    top_flows: np.ndarray

    def __init__(self, flow_field: np.ndarray, top_ratio: float=0.05):
        self.top_flows = flow_field
        

def _extract_top_flows(flow_field: np.ndarray, top_ratio: float) -> tuple[float, float]:
    u = flow_field[0].reshape(-1)
    v = flow_field[1].reshape(-1)
    mag = np.sqrt(u**2 + v**2)
    k = max(1, int(len(mag) * top_ratio))
    idx = np.argpartition(mag, -k)[-k:]
    top_u = u[idx]
    top_v = v[idx]
    return (np.mean(top_u), np.mean(top_v))

def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def get_motion_score(feat1: MotionFeature, feat2: MotionFeature) -> float:    
    f1 = feat1.top_flows
    f2 = feat2.top_flows

    m1 = np.mean(np.linalg.norm(f1, axis=2))  
    m2 = np.mean(np.linalg.norm(f2, axis=2))

    mag_sim = 40.0 / (np.abs(m1 - m2) + 40.0)

    dir_sim = np.dot(f1.flatten(), f2.flatten()) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    dir_sim = (dir_sim + 1) / 2

    dir_conf = (m1 / (m1 + 2)) * (m2 / (m2 + 2))

    return mag_sim * (1 - dir_conf) + dir_sim * dir_conf