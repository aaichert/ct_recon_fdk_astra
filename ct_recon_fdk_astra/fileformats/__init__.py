from .utils import (
    ctCircularTrajectoryToParameters,
    makeCircularTrajectory,
    camera_look_at
)
from .ompl import load_ompl, save_ompl
from .siemens import (
    load_projtable_xml,
    load_projtable_txt,
    save_projtable_txt,
    save_projtable_xml
)
from .stanford import load_conrad, save_conrad
from .rtk import load_rtk, save_rtk
from .astra import load_astra, save_astra

def discover_formats():
    """
    Returns registered loaders and savers for dynamic geometry format import/export.
    Returns:
        loaders: dict (format_name -> dict with "extensions" (list of str) and "fn" (callable))
        savers: dict (format_name -> dict with "extensions" (list of str) and "fn" (callable))
    """
    loaders = {
        "ompl": {
            "extensions": [".ompl"],
            "fn": load_ompl
        },
        "Siemens Projtable XML": {
            "extensions": [".xml"],
            "fn": load_projtable_xml
        },
        "Siemens Projtable TXT": {
            "extensions": [".txt"],
            "fn": load_projtable_txt
        },
        "Stanford CONRAD XML": {
            "extensions": [".xml"],
            "fn": load_conrad
        },
        "RTK Geometry XML": {
            "extensions": [".xml"],
            "fn": load_rtk
        },
        "ASTRA Vector Geometry": {
            "extensions": [".txt", ".vec"],
            "fn": load_astra
        }
    }
    savers = {
        "ompl": {
            "extensions": [".ompl"],
            "fn": save_ompl
        },
        "Siemens Projtable XML": {
            "extensions": [".xml"],
            "fn": save_projtable_xml
        },
        "Siemens Projtable TXT": {
            "extensions": [".txt"],
            "fn": save_projtable_txt
        },
        "Stanford CONRAD XML": {
            "extensions": [".xml"],
            "fn": save_conrad
        },
        "RTK Geometry XML": {
            "extensions": [".xml"],
            "fn": save_rtk
        },
        "ASTRA Vector Geometry": {
            "extensions": [".txt", ".vec"],
            "fn": save_astra
        }
    }
    return loaders, savers


__all__ = [
    "ctCircularTrajectoryToParameters",
    "makeCircularTrajectory",
    "camera_look_at",
    "load_ompl",
    "save_ompl",
    "load_projtable_xml",
    "load_projtable_txt",
    "save_projtable_txt",
    "save_projtable_xml",
    "load_conrad",
    "save_conrad",
    "load_rtk",
    "save_rtk",
    "load_astra",
    "save_astra",
    "discover_formats"
]
