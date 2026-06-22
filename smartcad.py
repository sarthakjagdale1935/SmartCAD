from __future__ import annotations

from io import StringIO
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRODUCT_TYPES = ["LIGHTING", "EV", "ADAS", "STRUCTURAL"]

PRODUCT_LABELS = {
    "LIGHTING": "Headlamp Housing",
    "EV": "PMSM Motor Housing",
    "ADAS": "ECU Bracket (ADAS)",
    "STRUCTURAL": "Structural Component",
}

FEATURE_COLUMNS = [
    "wall_thickness_mm",
    "fillet_radius_mm",
    "draft_angle_deg",
    "hole_diameter_mm",
    "rib_height_mm",
    "rib_thickness_mm",
    "tolerance_mm",
    "material_density",
    "surface_finish_ra",
    "assembly_clearance_mm",
    "overhang_angle_deg",
    "min_feature_size_mm",
    "aspect_ratio",
    "part_weight_kg",
]

RULE_ONLY_COLUMNS = [
    "vent_aspect_ratio",
    "cooling_channel_mm",
    "ip_groove_mm",
]

ALL_DESIGN_COLUMNS = FEATURE_COLUMNS + RULE_ONLY_COLUMNS

FEATURE_DEFINITIONS = [
    {"key": "wall_thickness_mm", "label": "Wall thickness", "unit": "mm", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.4, "group": "Geometry"},
    {"key": "fillet_radius_mm", "label": "Fillet radius", "unit": "mm", "min": 0.2, "max": 2.2, "step": 0.05, "default": 1.0, "group": "Geometry"},
    {"key": "draft_angle_deg", "label": "Draft angle", "unit": "deg", "min": 0.3, "max": 3.2, "step": 0.1, "default": 2.0, "group": "Geometry"},
    {"key": "hole_diameter_mm", "label": "Hole diameter", "unit": "mm", "min": 2.0, "max": 16.0, "step": 0.1, "default": 8.0, "group": "Geometry"},
    {"key": "rib_height_mm", "label": "Rib height", "unit": "mm", "min": 5.0, "max": 24.0, "step": 0.1, "default": 9.0, "group": "Ribs"},
    {"key": "rib_thickness_mm", "label": "Rib thickness", "unit": "mm", "min": 1.5, "max": 5.2, "step": 0.1, "default": 3.2, "group": "Ribs"},
    {"key": "tolerance_mm", "label": "Tolerance", "unit": "mm", "min": 0.02, "max": 0.2, "step": 0.005, "default": 0.05, "group": "Manufacturing"},
    {"key": "material_density", "label": "Material density", "unit": "g/cc", "min": 1.0, "max": 8.0, "step": 0.01, "default": 1.18, "group": "Material"},
    {"key": "surface_finish_ra", "label": "Surface finish Ra", "unit": "um", "min": 0.6, "max": 4.2, "step": 0.05, "default": 1.2, "group": "Manufacturing"},
    {"key": "assembly_clearance_mm", "label": "Assembly clearance", "unit": "mm", "min": 0.04, "max": 0.55, "step": 0.01, "default": 0.3, "group": "Assembly"},
    {"key": "overhang_angle_deg", "label": "Overhang angle", "unit": "deg", "min": 28.0, "max": 62.0, "step": 0.5, "default": 38.0, "group": "Manufacturing"},
    {"key": "min_feature_size_mm", "label": "Min feature size", "unit": "mm", "min": 0.25, "max": 2.5, "step": 0.05, "default": 1.5, "group": "Manufacturing"},
    {"key": "aspect_ratio", "label": "Part aspect ratio", "unit": ":1", "min": 2.0, "max": 9.5, "step": 0.1, "default": 3.2, "group": "Structure"},
    {"key": "part_weight_kg", "label": "Part weight", "unit": "kg", "min": 0.15, "max": 2.4, "step": 0.01, "default": 0.45, "group": "Material"},
    {"key": "vent_aspect_ratio", "label": "Vent aspect ratio", "unit": ":1", "min": 0.0, "max": 7.5, "step": 0.1, "default": 3.0, "group": "Product-specific"},
    {"key": "cooling_channel_mm", "label": "Cooling channel", "unit": "mm", "min": 0.0, "max": 5.8, "step": 0.1, "default": 0.0, "group": "Product-specific"},
    {"key": "ip_groove_mm", "label": "IP groove", "unit": "mm", "min": 0.0, "max": 2.1, "step": 0.1, "default": 0.0, "group": "Product-specific"},
]

_DATA_CSV = """wall_thickness_mm,fillet_radius_mm,draft_angle_deg,hole_diameter_mm,rib_height_mm,rib_thickness_mm,tolerance_mm,material_density,surface_finish_ra,assembly_clearance_mm,overhang_angle_deg,min_feature_size_mm,aspect_ratio,part_weight_kg,vent_aspect_ratio,cooling_channel_mm,ip_groove_mm,label,product_type
2.5,1.2,2.0,8.0,9.0,3.5,0.050,1.18,1.2,0.30,38.0,1.5,3.2,0.45,3.5,0.0,0.0,PASS,LIGHTING
1.4,0.6,0.8,5.0,14.0,2.0,0.120,1.20,2.8,0.10,52.0,0.8,6.5,0.72,5.5,0.0,0.0,FAIL,LIGHTING
3.0,1.5,2.5,10.0,8.0,3.0,0.040,1.15,0.9,0.35,35.0,2.0,2.8,0.38,3.2,0.0,0.0,PASS,LIGHTING
1.8,0.9,1.6,7.0,11.0,3.8,0.070,1.18,1.4,0.22,41.0,1.3,4.0,0.52,3.8,0.0,0.0,PASS,LIGHTING
1.2,0.4,0.6,3.5,20.0,2.5,0.150,1.22,3.5,0.08,58.0,0.5,8.2,0.88,6.0,0.0,0.0,FAIL,LIGHTING
2.8,1.4,2.2,9.5,8.5,3.2,0.040,1.16,1.0,0.32,36.0,1.8,3.0,0.41,3.3,0.0,0.0,PASS,LIGHTING
1.6,0.7,1.2,6.0,13.0,3.5,0.090,1.20,2.0,0.18,45.0,1.1,4.5,0.60,4.2,0.0,0.0,FAIL,LIGHTING
2.2,1.1,2.0,8.5,9.5,3.8,0.050,1.17,1.3,0.28,39.0,1.6,3.5,0.47,3.6,0.0,0.0,PASS,LIGHTING
1.0,0.3,0.5,2.5,22.0,2.0,0.180,1.24,4.0,0.06,60.0,0.4,9.0,0.92,7.0,0.0,0.0,FAIL,LIGHTING
3.5,2.0,3.0,12.0,7.0,3.5,0.030,1.14,0.8,0.40,32.0,2.2,2.5,0.33,2.8,0.0,0.0,PASS,LIGHTING
2.1,1.0,1.8,8.0,10.2,3.0,0.055,1.18,1.4,0.12,38.0,1.2,3.4,0.48,3.8,0.0,0.0,FAIL,LIGHTING
2.4,1.3,2.1,9.0,9.0,3.3,0.045,1.17,1.1,0.31,37.0,1.7,3.1,0.43,3.4,0.0,0.0,PASS,LIGHTING
1.3,0.5,0.9,4.0,18.0,2.2,0.130,1.21,3.0,0.09,55.0,0.6,7.5,0.80,5.8,0.0,0.0,FAIL,LIGHTING
2.6,1.4,2.3,9.2,8.8,3.4,0.042,1.16,1.05,0.33,36.5,1.75,3.0,0.42,3.35,0.0,0.0,PASS,LIGHTING
2.5,1.0,2.0,12.0,9.0,3.5,0.040,2.72,1.2,0.30,38.0,1.5,3.2,1.85,0.0,4.5,1.5,PASS,EV
1.4,0.6,1.6,10.0,9.0,3.5,0.050,2.70,1.0,0.25,40.0,1.2,3.8,1.90,0.0,2.1,1.5,FAIL,EV
3.0,1.2,2.5,14.0,8.0,4.0,0.030,2.75,0.9,0.35,35.0,2.0,2.8,2.20,0.0,5.0,1.8,PASS,EV
2.0,0.7,1.8,11.0,10.0,3.8,0.045,2.71,1.1,0.28,41.0,1.3,4.0,1.95,0.0,3.8,1.4,PASS,EV
1.2,0.4,0.9,8.0,12.0,3.5,0.120,2.68,2.5,0.15,50.0,0.8,6.5,2.10,0.0,1.8,0.8,FAIL,EV
2.8,1.1,2.2,13.0,8.5,4.2,0.035,2.73,0.95,0.32,37.0,1.8,3.0,2.05,0.0,4.2,1.6,PASS,EV
1.6,0.5,1.4,9.5,11.0,3.8,0.080,2.69,1.8,0.20,46.0,1.0,5.0,2.15,0.0,2.5,1.1,FAIL,EV
2.3,0.9,2.0,12.5,9.5,4.0,0.040,2.71,1.05,0.30,39.0,1.6,3.5,1.92,0.0,4.0,1.5,PASS,EV
1.0,0.3,0.7,7.0,14.0,3.5,0.150,2.65,3.0,0.10,54.0,0.6,7.8,2.25,0.0,1.5,0.7,FAIL,EV
3.5,1.5,3.0,15.0,7.5,4.5,0.025,2.76,0.8,0.40,32.0,2.2,2.5,1.75,0.0,5.5,2.0,PASS,EV
2.2,0.85,1.9,12.0,9.2,3.8,0.042,2.71,1.0,0.29,39.5,1.55,3.3,1.88,0.0,3.9,1.45,PASS,EV
1.5,0.55,1.3,9.0,11.5,3.7,0.090,2.69,2.0,0.18,47.0,0.9,5.5,2.12,0.0,2.3,1.0,FAIL,EV
2.5,0.9,2.0,6.0,7.5,3.0,0.060,1.05,1.5,0.28,35.0,0.8,2.8,0.22,0.0,0.0,1.4,PASS,ADAS
1.5,0.5,1.0,4.0,10.0,2.5,0.100,1.08,2.5,0.15,48.0,0.5,5.5,0.35,0.0,0.0,0.9,FAIL,ADAS
3.0,1.2,2.5,8.0,6.5,3.2,0.040,1.04,1.0,0.35,30.0,1.2,2.2,0.18,0.0,0.0,1.8,PASS,ADAS
2.0,0.8,1.8,5.5,8.0,3.5,0.070,1.06,1.8,0.24,38.0,0.9,3.5,0.26,0.0,0.0,1.3,PASS,ADAS
1.2,0.3,0.7,3.0,14.0,2.0,0.140,1.10,3.5,0.08,55.0,0.4,7.0,0.42,0.0,0.0,0.7,FAIL,ADAS
2.8,1.1,2.2,7.0,7.0,3.5,0.050,1.04,1.2,0.30,32.0,1.0,2.5,0.20,0.0,0.0,1.6,PASS,ADAS
1.8,0.7,1.5,5.0,9.0,3.0,0.080,1.07,2.0,0.20,42.0,0.7,4.0,0.30,0.0,0.0,1.1,PASS,ADAS
1.0,0.25,0.5,2.5,16.0,2.0,0.180,1.12,4.0,0.06,58.0,0.3,8.5,0.50,0.0,0.0,0.5,FAIL,ADAS
2.2,0.85,1.9,6.5,7.8,3.2,0.055,1.05,1.4,0.27,34.0,0.95,2.9,0.21,0.0,0.0,1.45,PASS,ADAS
1.4,0.45,0.9,3.8,12.0,2.2,0.120,1.09,3.0,0.10,50.0,0.45,6.0,0.38,0.0,0.0,0.85,FAIL,ADAS
2.5,1.0,2.0,10.0,10.0,4.0,0.060,1.55,1.8,0.30,38.0,1.5,3.0,0.65,0.0,0.0,0.0,PASS,STRUCTURAL
1.5,0.5,1.0,7.0,15.0,3.0,0.120,1.58,3.0,0.18,50.0,0.8,6.0,0.95,0.0,0.0,0.0,FAIL,STRUCTURAL
3.0,1.5,2.5,12.0,9.0,4.5,0.040,1.52,1.2,0.38,32.0,2.0,2.5,0.55,0.0,0.0,0.0,PASS,STRUCTURAL
2.2,1.0,1.8,9.5,11.0,4.2,0.070,1.54,2.0,0.26,40.0,1.4,3.8,0.72,0.0,0.0,0.0,PASS,STRUCTURAL
1.0,0.3,0.5,5.0,20.0,2.5,0.160,1.60,4.0,0.08,58.0,0.5,8.5,1.20,0.0,0.0,0.0,FAIL,STRUCTURAL
2.8,1.4,2.2,11.0,9.5,4.8,0.045,1.51,1.4,0.34,35.0,1.8,2.8,0.60,0.0,0.0,0.0,PASS,STRUCTURAL
1.8,0.7,1.4,8.0,13.0,4.0,0.090,1.56,2.5,0.20,44.0,1.1,4.5,0.82,0.0,0.0,0.0,FAIL,STRUCTURAL
3.5,2.0,3.0,14.0,8.0,5.0,0.030,1.50,0.9,0.42,30.0,2.4,2.2,0.48,0.0,0.0,0.0,PASS,STRUCTURAL
2.0,0.9,1.7,9.0,11.5,4.5,0.080,1.53,2.2,0.23,41.0,1.3,4.2,0.78,0.0,0.0,0.0,PASS,STRUCTURAL
1.2,0.4,0.8,6.0,18.0,3.0,0.140,1.59,3.5,0.10,54.0,0.6,7.2,1.05,0.0,0.0,0.0,FAIL,STRUCTURAL
2.3,1.1,2.0,8.5,9.5,3.6,0.050,1.17,1.25,0.29,37.5,1.65,3.2,0.44,3.6,0.0,0.0,PASS,LIGHTING
1.7,0.65,1.5,10.5,10.5,3.7,0.070,2.70,1.5,0.22,43.0,1.15,4.2,2.00,0.0,2.8,1.2,FAIL,EV
2.6,1.0,2.1,7.0,7.2,3.2,0.052,1.05,1.35,0.31,33.0,0.98,2.7,0.20,0.0,0.0,1.5,PASS,ADAS
2.1,0.88,1.78,9.2,11.8,4.3,0.075,1.53,2.1,0.24,42.0,1.28,4.0,0.75,0.0,0.0,0.0,FAIL,STRUCTURAL
3.2,1.8,2.8,11.0,7.5,3.8,0.035,1.14,0.85,0.38,33.5,2.1,2.6,0.36,3.1,0.0,0.0,PASS,LIGHTING
2.6,1.05,2.1,13.0,8.8,4.3,0.038,2.72,0.98,0.31,38.5,1.72,3.2,1.90,0.0,4.3,1.6,PASS,EV
1.3,0.42,0.85,3.5,13.5,2.2,0.130,1.09,3.2,0.09,52.0,0.42,6.5,0.40,0.0,0.0,0.75,FAIL,ADAS
3.2,1.6,2.8,12.5,8.5,4.8,0.032,1.51,1.05,0.40,31.0,2.2,2.4,0.52,0.0,0.0,0.0,PASS,STRUCTURAL
"""

DESIGN_RULES = {
    "R01": {
        "name": "Minimum Wall Thickness",
        "context": "Headlamp housing / Motor casing",
        "parameter": "wall_thickness_mm",
        "condition": lambda v: v >= 2.0,
        "severity": "CRITICAL",
        "message": "Wall {val:.2f}mm < 2.0mm - thermal/vibration failure risk.",
        "standard": "DFM-VAR-001",
        "applies_to": PRODUCT_TYPES,
    },
    "R02": {
        "name": "Thermal Vent Aspect Ratio",
        "context": "Headlamp condensate vents",
        "parameter": "vent_aspect_ratio",
        "condition": lambda v: (v <= 4.0) if v > 0 else True,
        "severity": "CRITICAL",
        "message": "Vent AR {val:.1f}:1 > 4:1 - moisture trap.",
        "standard": "LGT-VAR-001",
        "applies_to": ["LIGHTING"],
    },
    "R03": {
        "name": "Lens-Housing Assembly Clearance",
        "context": "Headlamp / DRL assembly stack",
        "parameter": "assembly_clearance_mm",
        "condition": lambda v: 0.20 <= v <= 0.50,
        "severity": "CRITICAL",
        "message": "Clearance {val:.2f}mm not in 0.20-0.50mm - lens fracture or IP breach.",
        "standard": "ASM-VAR-001",
        "applies_to": ["LIGHTING"],
    },
    "R04": {
        "name": "Draft Angle",
        "context": "Injection-moulded Varroc parts",
        "parameter": "draft_angle_deg",
        "condition": lambda v: v >= 1.5,
        "severity": "MAJOR",
        "message": "Draft {val:.1f}deg < 1.5deg - mold release failure.",
        "standard": "DFM-VAR-002",
        "applies_to": PRODUCT_TYPES,
    },
    "R05": {
        "name": "Fillet Radius",
        "context": "Motor housing / structural stress zones",
        "parameter": "fillet_radius_mm",
        "condition": lambda v: v >= 0.8,
        "severity": "MAJOR",
        "message": "Fillet {val:.2f}mm < 0.8mm - fatigue crack initiation.",
        "standard": "DFM-VAR-003",
        "applies_to": ["EV", "STRUCTURAL"],
    },
    "R06": {
        "name": "Cooling Channel Width",
        "context": "PMSM motor cooling jacket",
        "parameter": "cooling_channel_mm",
        "condition": lambda v: (v >= 3.0) if v > 0 else True,
        "severity": "CRITICAL",
        "message": "Cooling channel {val:.1f}mm < 3.0mm - demagnetisation risk.",
        "standard": "EV-VAR-001",
        "applies_to": ["EV"],
    },
    "R07": {
        "name": "Overhang Angle",
        "context": "Bracket / body structural / die-cast parts",
        "parameter": "overhang_angle_deg",
        "condition": lambda v: v <= 45.0,
        "severity": "MAJOR",
        "message": "Overhang {val:.1f}deg > 45deg - support structures required.",
        "standard": "STR-VAR-001",
        "applies_to": PRODUCT_TYPES,
    },
    "R08": {
        "name": "Minimum Feature Size",
        "context": "EV connector / PCB brackets / ADAS mounts",
        "parameter": "min_feature_size_mm",
        "condition": lambda v: v >= 0.6,
        "severity": "MAJOR",
        "message": "Feature {val:.2f}mm < 0.6mm - beyond tooling capability.",
        "standard": "MFG-VAR-001",
        "applies_to": ["EV", "ADAS", "STRUCTURAL"],
    },
    "R09": {
        "name": "Surface Finish - Sealing Surfaces",
        "context": "Headlamp / ECU sealing and mating surfaces",
        "parameter": "surface_finish_ra",
        "condition": lambda v: v <= 1.6,
        "severity": "MINOR",
        "message": "Ra {val:.1f}um > 1.6um - IP67 seal quality insufficient.",
        "standard": "SFC-VAR-001",
        "applies_to": ["LIGHTING", "ADAS"],
    },
    "R10": {
        "name": "Rib Height-to-Thickness Ratio",
        "context": "All injection-moulded Varroc parts",
        "parameter": None,
        "condition": None,
        "severity": "MAJOR",
        "message": "Rib H/T {val:.1f}:1 > 3:1 - sink marks on class-A surfaces.",
        "standard": "DFM-VAR-004",
        "applies_to": PRODUCT_TYPES,
        "custom": True,
    },
    "R11": {
        "name": "Part Aspect Ratio",
        "context": "Structural brackets / motor mounts",
        "parameter": "aspect_ratio",
        "condition": lambda v: v <= 5.0,
        "severity": "MAJOR",
        "message": "AR {val:.1f}:1 > 5:1 - NVH buckling risk.",
        "standard": "STR-VAR-002",
        "applies_to": ["STRUCTURAL", "EV"],
    },
    "R12": {
        "name": "IP Rating Sealing Groove",
        "context": "Sensor / ECU / connector enclosures",
        "parameter": "ip_groove_mm",
        "condition": lambda v: (v >= 1.2) if v > 0 else True,
        "severity": "MAJOR",
        "message": "IP groove {val:.1f}mm < 1.2mm - O-ring will not fit, IP67 fails.",
        "standard": "IP-VAR-001",
        "applies_to": ["ADAS", "EV"],
    },
}

SAMPLE_CASES = [
    {
        "design_id": "CASE A",
        "product_type": "LIGHTING",
        "description": "Headlamp housing with tight assembly clearance",
        "expected_verdict": "WARNING",
        "features": {
            "wall_thickness_mm": 2.1,
            "fillet_radius_mm": 1.1,
            "draft_angle_deg": 2.0,
            "hole_diameter_mm": 8.0,
            "rib_height_mm": 9.0,
            "rib_thickness_mm": 3.0,
            "tolerance_mm": 0.05,
            "material_density": 1.18,
            "surface_finish_ra": 1.2,
            "assembly_clearance_mm": 0.12,
            "overhang_angle_deg": 38.0,
            "min_feature_size_mm": 1.5,
            "aspect_ratio": 3.2,
            "part_weight_kg": 0.45,
            "vent_aspect_ratio": 3.0,
            "cooling_channel_mm": 0.0,
            "ip_groove_mm": 0.0,
        },
    },
    {
        "design_id": "CASE B",
        "product_type": "EV",
        "description": "PMSM motor housing with thin wall and narrow cooling channel",
        "expected_verdict": "FAIL",
        "features": {
            "wall_thickness_mm": 1.4,
            "fillet_radius_mm": 0.7,
            "draft_angle_deg": 1.8,
            "hole_diameter_mm": 11.0,
            "rib_height_mm": 9.5,
            "rib_thickness_mm": 3.8,
            "tolerance_mm": 0.045,
            "material_density": 2.71,
            "surface_finish_ra": 1.1,
            "assembly_clearance_mm": 0.28,
            "overhang_angle_deg": 41.0,
            "min_feature_size_mm": 1.3,
            "aspect_ratio": 4.0,
            "part_weight_kg": 1.95,
            "vent_aspect_ratio": 0.0,
            "cooling_channel_mm": 2.1,
            "ip_groove_mm": 1.4,
        },
    },
    {
        "design_id": "CASE C",
        "product_type": "ADAS",
        "description": "ADAS ECU bracket with all listed specs met",
        "expected_verdict": "PASS",
        "features": {
            "wall_thickness_mm": 2.6,
            "fillet_radius_mm": 1.0,
            "draft_angle_deg": 2.1,
            "hole_diameter_mm": 7.0,
            "rib_height_mm": 7.2,
            "rib_thickness_mm": 3.2,
            "tolerance_mm": 0.052,
            "material_density": 1.05,
            "surface_finish_ra": 1.35,
            "assembly_clearance_mm": 0.31,
            "overhang_angle_deg": 33.0,
            "min_feature_size_mm": 0.98,
            "aspect_ratio": 2.7,
            "part_weight_kg": 0.20,
            "vent_aspect_ratio": 0.0,
            "cooling_channel_mm": 0.0,
            "ip_groove_mm": 1.5,
        },
    },
]


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(StringIO(_DATA_CSV))
    df["label_enc"] = (df["label"] == "PASS").astype(int)
    return df


def _round_float(value: float, ndigits: int = 4) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(float(value), ndigits)


def train_model() -> tuple[Pipeline, dict[str, Any]]:
    df = build_dataset()
    x = df[FEATURE_COLUMNS].values
    y = df["label_enc"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=4,
                    min_samples_split=3,
                    subsample=0.85,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="accuracy")

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = 0.0

    classifier = pipeline.named_steps["classifier"]
    importances = {
        feature: round(float(value), 5)
        for feature, value in sorted(
            zip(FEATURE_COLUMNS, classifier.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    model_card = {
        "name": "SmartCAD-AI GBM",
        "source": "SmartCAD_AI.ipynb",
        "problem": "AI-driven CAD design validation",
        "algorithm": "GradientBoostingClassifier",
        "dataset_size": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "test_accuracy": _round_float(accuracy_score(y_test, y_pred)),
        "f1_score": _round_float(f1_score(y_test, y_pred)),
        "cv_mean": _round_float(cv_scores.mean()),
        "cv_std": _round_float(cv_scores.std()),
        "roc_auc": _round_float(roc_auc),
        "varroc_rules": list(DESIGN_RULES.keys()),
        "product_lines": PRODUCT_TYPES,
        "feature_importances": importances,
    }
    return pipeline, model_card


GBM_PIPELINE, MODEL_CARD = train_model()


def normalize_features(features: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in ALL_DESIGN_COLUMNS:
        raw_value = features.get(key, 0.0)
        try:
            normalized[key] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
    return normalized


def run_rule_engine(design: dict[str, float], product_type: str | None = None) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    passed: list[dict[str, str]] = []
    skipped: list[str] = []

    for rule_id, rule in DESIGN_RULES.items():
        if product_type and product_type not in rule.get("applies_to", []):
            skipped.append(rule_id)
            continue

        if rule.get("custom"):
            ratio = design["rib_height_mm"] / max(design["rib_thickness_mm"], 0.01)
            entry = {
                "rule_id": rule_id,
                "rule_name": rule["name"],
                "context": rule["context"],
                "severity": rule["severity"],
                "message": rule["message"].format(val=ratio),
                "standard": rule["standard"],
                "measured_value": round(ratio, 2),
            }
            if ratio > 3.0:
                violations.append(entry)
            else:
                passed.append({"rule_id": rule_id, "rule_name": rule["name"]})
            continue

        parameter = rule["parameter"]
        value = design.get(parameter)
        if value is None:
            skipped.append(rule_id)
            continue

        entry = {
            "rule_id": rule_id,
            "rule_name": rule["name"],
            "context": rule["context"],
            "severity": rule["severity"],
            "message": rule["message"].format(val=value),
            "standard": rule["standard"],
            "measured_value": value,
        }

        if not rule["condition"](value):
            violations.append(entry)
        else:
            passed.append({"rule_id": rule_id, "rule_name": rule["name"]})

    critical_count = sum(1 for v in violations if v["severity"] == "CRITICAL")
    major_count = sum(1 for v in violations if v["severity"] == "MAJOR")
    minor_count = sum(1 for v in violations if v["severity"] == "MINOR")

    if critical_count > 0:
        verdict = "FAIL"
        risk = "HIGH"
    elif major_count >= 2:
        verdict = "FAIL"
        risk = "MEDIUM-HIGH"
    elif major_count == 1:
        verdict = "WARNING"
        risk = "MEDIUM"
    else:
        verdict = "PASS"
        risk = "LOW"

    return {
        "violations": violations,
        "passed_rules": passed,
        "skipped_rules": skipped,
        "rule_verdict": verdict,
        "risk_level": risk,
        "critical_count": critical_count,
        "major_count": major_count,
        "minor_count": minor_count,
    }


def ml_predict(design: dict[str, float]) -> dict[str, Any]:
    feature_vector = np.array([[design[column] for column in FEATURE_COLUMNS]])
    prediction = int(GBM_PIPELINE.predict(feature_vector)[0])
    probabilities = GBM_PIPELINE.predict_proba(feature_vector)[0]
    return {
        "ml_verdict": "PASS" if prediction == 1 else "FAIL",
        "pass_prob": round(float(probabilities[1]) * 100, 1),
        "fail_prob": round(float(probabilities[0]) * 100, 1),
        "confidence": round(float(max(probabilities)) * 100, 1),
    }


def fuse_verdicts(rule_result: dict[str, Any], ml_result: dict[str, Any]) -> dict[str, Any]:
    rule_verdict = rule_result["rule_verdict"]
    ml_verdict = ml_result["ml_verdict"]
    ml_confidence = ml_result["confidence"]

    if rule_result["critical_count"] > 0:
        return {
            "verdict": "FAIL",
            "method": "Rule override - critical Varroc safety violation",
            "confidence": 99.0,
        }

    if rule_verdict == "PASS" and ml_verdict == "PASS":
        return {
            "verdict": "PASS",
            "method": "Consensus - Rule Engine + GBM agree PASS",
            "confidence": round(min(99, (ml_confidence + 92) / 2), 1),
        }

    if rule_verdict in ("FAIL", "WARNING") and ml_verdict == "FAIL":
        return {
            "verdict": "FAIL",
            "method": "Consensus - Rule Engine + GBM agree FAIL",
            "confidence": round(min(99, (ml_confidence + 88) / 2), 1),
        }

    if rule_verdict == "PASS" and ml_verdict == "FAIL":
        if ml_confidence > 75:
            return {
                "verdict": "FAIL",
                "method": "GBM override - interaction pattern beyond explicit rules",
                "confidence": ml_confidence,
            }
        return {
            "verdict": "WARNING",
            "method": "Conflict - escalate to Varroc design lead",
            "confidence": ml_confidence,
        }

    if rule_verdict in ("FAIL", "WARNING") and ml_verdict == "PASS":
        return {
            "verdict": "WARNING",
            "method": "Partial flag - rule concern, GBM uncertain; manual review",
            "confidence": ml_confidence,
        }

    return {
        "verdict": "WARNING",
        "method": "Ambiguous - review recommended",
        "confidence": 60.0,
    }


def validate_design(
    features: dict[str, Any],
    product_type: str,
    design_id: str = "CUSTOM-001",
    description: str | None = None,
) -> dict[str, Any]:
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"product_type must be one of {', '.join(PRODUCT_TYPES)}")

    design = normalize_features(features)
    rule_result = run_rule_engine(design, product_type=product_type)
    ml_result = ml_predict(design)
    fusion = fuse_verdicts(rule_result, ml_result)

    return {
        "design_id": design_id,
        "product_type": product_type,
        "product_label": PRODUCT_LABELS[product_type],
        "description": description,
        "features": design,
        "rule_result": rule_result,
        "ml_result": ml_result,
        "fusion": fusion,
    }


def dataset_summary() -> dict[str, Any]:
    df = build_dataset()
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "labels": {key: int(value) for key, value in df["label"].value_counts().items()},
        "product_types": {
            key: int(value) for key, value in df["product_type"].value_counts().items()
        },
    }


def metadata() -> dict[str, Any]:
    return {
        "product_types": PRODUCT_TYPES,
        "product_labels": PRODUCT_LABELS,
        "feature_columns": FEATURE_COLUMNS,
        "rule_only_columns": RULE_ONLY_COLUMNS,
        "feature_definitions": FEATURE_DEFINITIONS,
        "rules": [
            {
                "rule_id": rule_id,
                "name": rule["name"],
                "severity": rule["severity"],
                "standard": rule["standard"],
                "applies_to": rule["applies_to"],
                "context": rule["context"],
            }
            for rule_id, rule in DESIGN_RULES.items()
        ],
        "samples": SAMPLE_CASES,
        "dataset": dataset_summary(),
        "model_card": MODEL_CARD,
    }

