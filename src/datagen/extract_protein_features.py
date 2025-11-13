#!/usr/bin/env python3
"""
Protein Feature Extraction Tool

This script extracts features from protein PDB files for use in machine learning models.
It generates SAS (Solvent Accessible Surface) points and calculates various chemical and
geometrical features for each point, similar to those found in P2Rank's vectorsTrain.csv.

Features include:
- SAS point coordinates (x, y, z)
- Chain ID
- Residue number
- Chemical properties (hydrophobicity, etc.)
- Atom density features
- Volumetric site features
- Protrusion
- B-factor

Usage:
    python extract_protein_features.py <dataset_file> <output_dir>

Example:
    python extract_protein_features.py ../p2rank-datasets/chen11.ds ./output

Use pocknet_env from conda to run this script:
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import concurrent.futures
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ProteinFeatureExtractor")

# ---------------------------------------------------------------------------
# Parameter defaults below are aligned with the official P2Rank implementation
# (see p2rank/distro/config/default.groovy).  Notably:
#   * neighbourhood_radius  -> 6.0 Å
#   * solvent probe radius  -> 1.6 Å
#   * tessellation level 2  -> approximated here by point_density=3.0 or
#                              sr_n_points=100 when using BioPython
#   * atom_table_feat_pow   -> 2 with sign discarded during exponentiation
# These settings ensure feature vectors match P2Rank defaults.

# BioPython imports for SASA calculation
try:
    from Bio.PDB import PDBParser, Selection, NeighborSearch
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure
    import Bio.PDB.ResidueDepth as ResidueDepth
    BIOPYTHON_AVAILABLE = True
except ImportError:
    logger.warning("BioPython not available. SASA calculation will use the custom implementation.")
    BIOPYTHON_AVAILABLE = False

# P2Rank canonical amino acids (matching ChemDefaults.groovy)
CANONICAL_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", 
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "STP"
}

# Constants
ATOM_TYPES = {
    'C': {'radius': 1.7, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 0.5},
    'N': {'radius': 1.55, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': -3.5},
    'O': {'radius': 1.52, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': -3.5},
    'S': {'radius': 1.8, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 0.2},
    'SE': {'radius': 1.8, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 0.2},  # Selenocysteine
    'P': {'radius': 1.8, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': -3.5},
    'H': {'radius': 1.2, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': -1.0},
    'F': {'radius': 1.47, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 2.5},
    'CL': {'radius': 1.75, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 1.0},
    'BR': {'radius': 1.85, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 1.0},
    'I': {'radius': 1.98, 'hydrophobic': True, 'hydrophilic': False, 'hydropathy': 1.0},
    'ZN': {'radius': 1.39, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': 0.0},
    'FE': {'radius': 1.32, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': 0.0},
    'CA': {'radius': 1.95, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': 0.0},
    'MG': {'radius': 1.37, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': 0.0},
    'NA': {'radius': 1.86, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': 0.0},
    'K': {'radius': 2.27, 'hydrophobic': False, 'hydrophilic': True, 'hydropathy': 0.0},
}

# Amino acid properties
AA_PROPERTIES = {
    'ALA': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 1.8, 'aliphatic': 1.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'ARG': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -4.5, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 1.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 1.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 1.0},
    'ASN': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -3.5, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 1.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 1.0, 'polar': 1.0, 'ionizable': 0.0},
    'ASP': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -3.5, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 1.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 1.0, 'h_donor': 0.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 1.0},
    'CYS': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 2.5, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 1.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 0.5},
    'GLN': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -3.5, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 1.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 1.0, 'polar': 1.0, 'ionizable': 0.0},
    'GLU': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -3.5, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 1.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 1.0, 'h_donor': 0.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 1.0},
    'GLY': {'hydrophobic': 0.5, 'hydrophilic': 0.5, 'hydropathy': -0.4, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'HIS': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -3.2, 'aliphatic': 0.0, 'aromatic': 1.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 1.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.5, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 1.0},
    'ILE': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 4.5, 'aliphatic': 1.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'LEU': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 3.8, 'aliphatic': 1.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'LYS': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -3.9, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 1.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 1.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 1.0},
    'MET': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 1.9, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 1.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'PHE': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 2.8, 'aliphatic': 0.0, 'aromatic': 1.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'PRO': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': -1.6, 'aliphatic': 1.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
    'SER': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -0.8, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 1.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 1.0, 'polar': 1.0, 'ionizable': 0.0},
    'THR': {'hydrophobic': 0.0, 'hydrophilic': 1.0, 'hydropathy': -0.7, 'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 1.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 1.0, 'polar': 1.0, 'ionizable': 0.0},
    'TRP': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': -0.9, 'aliphatic': 0.0, 'aromatic': 1.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.5, 'ionizable': 0.0},
    'TYR': {'hydrophobic': 0.5, 'hydrophilic': 0.5, 'hydropathy': -1.3, 'aliphatic': 0.0, 'aromatic': 1.0, 'sulfur': 0.0, 'hydroxyl': 1.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 1.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 1.0, 'polar': 1.0, 'ionizable': 0.5},
    'VAL': {'hydrophobic': 1.0, 'hydrophilic': 0.0, 'hydropathy': 4.2, 'aliphatic': 1.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0},
}

# Residue mapping for non-standard amino acids
RESIDUE_MAPPING = {
    'MSE': 'MET',  # Selenomethionine -> Methionine (biochemically justified)
    'SEC': 'CYS',  # Selenocysteine -> Cysteine (if present)
    'PYL': 'LYS',  # Pyrrolysine -> Lysine (if present)
}

# Properties for non-standard residues that can't be mapped
NON_STANDARD_PROPERTIES = {
    'HEM': {  # Heme group - iron-containing porphyrin
        'hydrophobic': 0.3, 'hydrophilic': 0.7, 'hydropathy': -2.0, 
        'aliphatic': 0.0, 'aromatic': 1.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 
        'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 
        'h_donor': 0.0, 'h_acceptor': 1.0, 'h_donor_acceptor': 0.0, 'polar': 1.0, 'ionizable': 0.0
    },
    'UNK': {  # Unknown residue - set to neutral/zero
        'hydrophobic': 0.0, 'hydrophilic': 0.0, 'hydropathy': 0.0, 
        'aliphatic': 0.0, 'aromatic': 0.0, 'sulfur': 0.0, 'hydroxyl': 0.0, 
        'basic': 0.0, 'acidic': 0.0, 'amide': 0.0, 'pos_charge': 0.0, 'neg_charge': 0.0, 
        'h_donor': 0.0, 'h_acceptor': 0.0, 'h_donor_acceptor': 0.0, 'polar': 0.0, 'ionizable': 0.0
    }
}

def get_residue_properties(residue_name):
    """Get properties for a residue, handling non-standard residues via mapping."""
    # First check if it's a standard amino acid
    if residue_name in AA_PROPERTIES:
        return AA_PROPERTIES[residue_name]
    
    # Check if it can be mapped to a standard amino acid
    if residue_name in RESIDUE_MAPPING:
        mapped_residue = RESIDUE_MAPPING[residue_name]
        logger.debug(f"Mapping {residue_name} -> {mapped_residue}")
        return AA_PROPERTIES[mapped_residue]
    
    # Check if it has specific non-standard properties
    if residue_name in NON_STANDARD_PROPERTIES:
        return NON_STANDARD_PROPERTIES[residue_name]
    
    # If all else fails, treat as unknown
    logger.warning(f"Unknown residue {residue_name}, treating as UNK")
    return NON_STANDARD_PROPERTIES['UNK']

# ===========================================================================
# P2Rank feature tables and chemistry defaults
# ---------------------------------------------------------------------------
# These tables are loaded from the CSV files shipped with the original P2Rank
# project. They are required in order to compute features identical to P2Rank.

CHEM_HEADER = [
    "hydrophobic", "hydrophilic", "hydrophatyIndex", "aliphatic", "aromatic",
    "sulfur", "hydroxyl", "basic", "acidic", "amide", "posCharge",
    "negCharge", "hBondDonor", "hBondAcceptor", "hBondDonorAcceptor",
    "polar", "ionizable", "atoms", "atomDensity", "atomC", "atomO", "atomN",
    "hDonorAtoms", "hAcceptorAtoms"
]

VOLSITE_HEADER = [
    "vsAromatic", "vsCation", "vsAnion", "vsHydrophobic", "vsAcceptor", "vsDonor"
]

ATOM_TABLE_HEADER = ["apRawValids", "apRawInvalids", "atomicHydrophobicity"]

# --- Chemical defaults (mirroring ChemDefaults.groovy) ---
HYDROPHOBIC = {
    "PHE":1, "ILE":1, "TRP":1, "GLY":1, "LEU":1, "VAL":1, "MET":1,
    "ALA":1, "CYS":1, "TYR":1,
    "ARG":-1, "ASN":-1, "ASP":-1, "GLN":-1, "GLU":-1, "LYS":-1, "PRO":-1
}

HYDROPHATY_INDEX = {
    "ALA":1.8, "ARG":-4.5, "ASN":-3.5, "ASP":-3.5, "CYS":2.5, "GLU":-3.5,
    "GLN":-3.5, "GLY":-0.4, "HIS":-3.2, "ILE":4.5, "LEU":3.8, "LYS":-3.9,
    "MET":1.9, "PHE":2.8, "PRO":-1.6, "SER":-0.8, "THR":-0.7, "TRP":-0.9,
    "TYR":-1.3, "VAL":4.2
}

ALIPHATIC = {"ALA":1, "LEU":1, "ILE":1, "VAL":1, "GLY":1, "PRO":1}
AROMATIC = {"PHE":1, "TRP":1, "TYR":1}
SULFUR = {"CYS":1, "MET":1}
HYDROXYL = {"SER":1, "THR":1}
BASIC = {"ARG":3, "LYS":2, "HIS":1}
ACIDIC = {"ASP":1, "GLU":1}
AMIDE = {"ASN":1, "GLN":1}
CHARGE = {"ASP":-1, "GLU":-1, "ARG":1, "HIS":1, "LYS":1}
POLAR = {"ARG":1,"ASN":1,"ASP":1,"GLN":1,"GLU":1,"HIS":1,"LYS":1,"SER":1,
         "THR":1,"TYR":1,"CYS":1}
IONIZABLE = {"ASP":1,"GLU":1,"HIS":1,"LYS":1,"ARG":1,"CYS":1,"TYR":1}
HB_DONOR = {"ARG":1,"LYS":1,"TRY":1}
HB_ACCEPTOR = {"ASP":1,"GLU":1}
HB_DONOR_ACCEPTOR = {"ASN":1,"GLN":1,"HIS":1,"SER":1,"THR":1,"TYR":1}

AACODES = [
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","STP"
]

def _aa_default(code):
    vec = {h:0.0 for h in CHEM_HEADER}
    vec["hydrophobic"] = max(HYDROPHOBIC.get(code,0),0)
    vec["hydrophilic"] = max(-HYDROPHOBIC.get(code,0),0)
    vec["hydrophatyIndex"] = HYDROPHATY_INDEX.get(code,0.0)
    vec["aliphatic"] = ALIPHATIC.get(code,0)
    vec["aromatic"] = AROMATIC.get(code,0)
    vec["sulfur"] = SULFUR.get(code,0)
    vec["hydroxyl"] = HYDROXYL.get(code,0)
    vec["basic"] = BASIC.get(code,0)
    vec["acidic"] = ACIDIC.get(code,0)
    vec["amide"] = AMIDE.get(code,0)
    charge = CHARGE.get(code,0)
    if charge >= 0:
        vec["posCharge"] = charge
    else:
        vec["negCharge"] = -charge
    vec["hBondDonor"] = HB_DONOR.get(code,0)
    vec["hBondAcceptor"] = HB_ACCEPTOR.get(code,0)
    vec["hBondDonorAcceptor"] = HB_DONOR_ACCEPTOR.get(code,0)
    vec["polar"] = POLAR.get(code,0)
    vec["ionizable"] = IONIZABLE.get(code,0)
    return vec

AA_DEFAULTS = {code:_aa_default(code) for code in AACODES}

ATOMIC_TABLE = {}
VOLSITE_TABLE = {}

def load_property_tables():
    """Load atomic-properties.csv and volsite-atomic-properties.csv."""
    base_dir = os.path.dirname(__file__)
    atomic_path = os.path.join(base_dir, "atomic-properties.csv")
    volsite_path = os.path.join(base_dir, "volsite-atomic-properties.csv")
    if not ATOMIC_TABLE:
        import csv
        with open(atomic_path, newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                name = row["atomName"].strip()
                ATOMIC_TABLE[name] = {
                    k: float(row[k]) for k in ATOM_TABLE_HEADER if k in row
                }
    if not VOLSITE_TABLE:
        import csv
        with open(volsite_path, newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                name = row["atomName"].strip()
                VOLSITE_TABLE[name] = {
                    "vsAromatic": float(row["vsAromatic"]),
                    "vsCation": float(row["vsCation"]),
                    "vsAnion": float(row["vsAnion"]),
                    "vsHydrophobic": float(row["vsHydrophobic"]),
                    "vsAcceptor": float(row["vsAcceptor"]),
                    "vsDonor": float(row["vsDonor"])
                }

load_property_tables()


def volsite_atom_properties(atom_name: str, residue_name: str):
    """Return VolSite pharmacophore flags for given atom/residue."""
    key = f"{residue_name}.{atom_name}"
    props = VOLSITE_TABLE.get(key)
    if props is not None:
        return props

    name = atom_name.upper()
    res = residue_name.upper() if residue_name else ""

    atm = {
        "vsAromatic": 0.0,
        "vsCation": 0.0,
        "vsAnion": 0.0,
        "vsHydrophobic": 0.0,
        "vsAcceptor": 0.0,
        "vsDonor": 0.0,
    }

    def set_prop(prop):
        atm[prop] = 1.0
        return atm

    if name == "C":
        return set_prop("vsHydrophobic")
    if name in {"CA", "CB"} and res != "CA":
        return set_prop("vsHydrophobic")
    if name == "CD" and res in {"ARG", "GLN", "GLU", "LYS", "PRO"}:
        return set_prop("vsHydrophobic")
    if name == "CD1":
        if res in {"ILE", "LEU"}:
            return set_prop("vsHydrophobic")
        elif res in {"PHE", "TRP", "TYR"}:
            return set_prop("vsAromatic")
    if name == "CD2":
        if res == "LEU":
            return set_prop("vsHydrophobic")
        elif res in {"PHE", "TRP", "TYR"}:
            return set_prop("vsAromatic")
        elif res in {"HIS", "HID", "HIE"}:
            atm["vsAromatic"] = 1.0
            atm["vsHydrophobic"] = 0.0
            return atm
    if name == "CE" and res in {"LYS", "MET"}:
        return set_prop("vsHydrophobic")
    if name == "CE1":
        if res in {"HIS", "HID", "HIE"}:
            atm["vsAromatic"] = 1.0
            atm["vsHydrophobic"] = 0.0
            return atm
        elif res in {"PHE", "TYR"}:
            return set_prop("vsAromatic")
    if name == "CE2" and res in {"PHE", "TRP", "TYR"}:
        return set_prop("vsAromatic")
    if name == "CE3" and res == "TRP":
        return set_prop("vsAromatic")
    if name == "CG":
        if res in {"ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "LEU", "LYS", "MET", "PRO"}:
            return set_prop("vsHydrophobic")
        elif res in {"HIS", "HID", "HIE", "PHE", "TRP", "TYR"}:
            return set_prop("vsAromatic")
    if (name == "CG1" and res in {"ILE", "VAL"}) or (name == "CG2" and res in {"ILE", "THR", "VAL"}):
        return set_prop("vsHydrophobic")
    if name == "CH2" and res == "TRP":
        return set_prop("vsAromatic")
    if name == "CZ":
        if res == "ARG":
            return set_prop("vsHydrophobic")
        elif res in {"PHE", "TYR"}:
            return set_prop("vsAromatic")
    if name in {"CZ2", "CZ3"} and res == "TRP":
        return set_prop("vsAromatic")
    if name == "N":
        return set_prop("vsDonor")
    if name == "ND1" and res in {"HIS", "HID", "HIE"}:
        atm["vsDonor"] = 1.0
        atm["vsAcceptor"] = 1.0
        return atm
    if name == "ND2" and res == "ASN":
        return set_prop("vsDonor")
    if name == "NE" and res in {"ARG", "LYS"}:
        return set_prop("vsCation")
    if name == "NE1" and res == "TRP":
        return set_prop("vsDonor")
    if name == "NE2":
        if res == "GLN":
            return set_prop("vsDonor")
        elif res in {"HIS", "HID", "HIE"}:
            atm["vsDonor"] = 1.0
            atm["vsAcceptor"] = 1.0
            return atm
    if (name in {"NH1", "NH2"} and res == "ARG") or (name == "NZ" and res == "LYS"):
        return set_prop("vsCation")
    if name == "O":
        return set_prop("vsAcceptor")
    if name in {"OD1", "OD2"} and res == "ASP":
        return set_prop("vsAnion")
    if name == "OD1" and res == "ASN":
        return set_prop("vsAcceptor")
    if name == "OE1" and res == "GLN":
        return set_prop("vsAcceptor")
    if name in {"OE1", "OE2"} and res == "GLU":
        return set_prop("vsAnion")
    if (name == "OG" and res == "SER") or (name == "OG1" and res == "THR") or (name == "OH" and res == "TYR"):
        atm["vsDonor"] = 1.0
        atm["vsAcceptor"] = 1.0
        return atm
    if name == "OXT":
        return set_prop("vsAnion")
    if name in {"SD", "SG"}:
        if res in {"CYS", "CYX"}:
            return set_prop("vsAcceptor")
        elif res == "MET":
            return set_prop("vsHydrophobic")
    if name in {"FE", "MG", "MN", "ZN", "CO"}:
        return set_prop("vsCation")

    return atm


# Donor/acceptor atoms (partial)
DONOR_ATOMS = {'N', 'NH1', 'NH2', 'NE', 'NE1', 'NE2', 'ND1', 'ND2', 'NZ', 'OG', 'OG1', 'OH'}
ACCEPTOR_ATOMS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'NE2', 'ND1', 'ND2', 'SG'}


class Atom:
    """Class representing an atom in a protein structure."""
    
    def __init__(self, atom_id, atom_name, residue_name, chain_id, residue_number,
                 x, y, z, occupancy, temp_factor, element):
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.residue_name = residue_name
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = occupancy
        self.temp_factor = temp_factor
        self.element = element
        
        # Determine atom properties
        self.is_donor = atom_name in DONOR_ATOMS
        self.is_acceptor = atom_name in ACCEPTOR_ATOMS
        self.is_backbone = atom_name in {'N', 'CA', 'C', 'O'}
        self.is_sidechain = not self.is_backbone
        
        # Get atom radius and properties from the ATOM_TYPES dictionary
        self.properties = ATOM_TYPES.get(element, {'radius': 1.7, 'hydrophobic': False, 'hydrophilic': False, 'hydropathy': 0.0})
        self.radius = self.properties['radius']
        self.hydrophobic = self.properties['hydrophobic']
        self.hydrophilic = self.properties['hydrophilic']
        self.hydropathy = self.properties['hydropathy']
        
        # Determine charge based on atom type and residue
        self.is_positively_charged = False
        self.is_negatively_charged = False
        
        # Assign charges based on residue type and atom name
        if residue_name == 'ARG' and atom_name in {'NH1', 'NH2', 'NE'}:
            self.is_positively_charged = True
        elif residue_name == 'LYS' and atom_name == 'NZ':
            self.is_positively_charged = True
        elif residue_name == 'HIS' and atom_name in {'ND1', 'NE2'}:
            self.is_positively_charged = True  # Partially charged
        elif residue_name == 'ASP' and atom_name in {'OD1', 'OD2'}:
            self.is_negatively_charged = True
        elif residue_name == 'GLU' and atom_name in {'OE1', 'OE2'}:
            self.is_negatively_charged = True
        
        # Determine if the atom is in an aromatic residue
        self.is_aromatic = residue_name in {'PHE', 'TYR', 'TRP', 'HIS'} and element == 'C'
        
    def get_coord(self):
        """Get the coordinates of the atom as a numpy array."""
        return np.array([self.x, self.y, self.z])
    
    def distance(self, other):
        """Calculate the distance to another atom or point."""
        if isinstance(other, Atom):
            other_coords = other.get_coord()
        else:  # Assume it's a numpy array or similar
            other_coords = other
        return np.linalg.norm(self.get_coord() - other_coords)


def surface_chem_vector(atom):
    """Compute chemical feature vector for an atom using the P2Rank heuristics."""
    res = atom.residue_name
    base = AA_DEFAULTS.get(res, {h:0.0 for h in CHEM_HEADER})
    vec = base.copy()

    # defaults for atom related counts
    vec["atoms"] = 1
    vec["atomDensity"] = 1
    vec["atomC"] = 0
    vec["atomO"] = 0
    vec["atomN"] = 0
    vec["hDonorAtoms"] = 0
    vec["hAcceptorAtoms"] = 0

    an = atom.atom_name

    if res == "ARG" and an in {"NE","NH1","NH2"}:
        vec["hDonorAtoms"] += 1
    elif res == "ASN":
        if an == "ND2":
            vec["hDonorAtoms"] += 1
        elif an == "OD1":
            vec["hAcceptorAtoms"] += 1
    elif res == "ASP" and an in {"OD1","OD2"}:
        vec["hAcceptorAtoms"] += 1
    elif res == "GLN":
        if an == "NE2":
            vec["hDonorAtoms"] += 1
        elif an == "OE1":
            vec["hAcceptorAtoms"] += 1
    elif res == "GLU" and an in {"OE1","OE2"}:
        vec["hAcceptorAtoms"] += 1
    elif res == "HIS" and an in {"ND1","NE2"}:
        vec["hDonorAtoms"] += 1
        vec["hAcceptorAtoms"] += 1
    elif res == "LYS" and an == "NZ":
        vec["hDonorAtoms"] += 1
    elif res == "SER" and an == "OG":
        vec["hDonorAtoms"] += 1
        vec["hAcceptorAtoms"] += 1
    elif res == "THR" and an == "OG1":
        vec["hDonorAtoms"] += 1
        vec["hAcceptorAtoms"] += 1
    elif res == "TRP" and an == "NE1":
        vec["hDonorAtoms"] += 1
    elif res == "TYR" and an == "OH":
        vec["hDonorAtoms"] += 1
        vec["hAcceptorAtoms"] += 1

    # element based counters
    if atom.element == "C" or an.startswith("C"):
        vec["atomC"] = 1
    elif atom.element == "O" or an.startswith("O"):
        vec["atomO"] = 1
    elif atom.element == "N" or an.startswith("N"):
        vec["atomN"] = 1

    return vec


class SASPoint:
    """
    Class representing a Solvent Accessible Surface point.
    
    A SAS point is a point on the surface of a protein that is accessible to solvent.
    It has coordinates, a normal vector pointing outward from the protein surface,
    and can be associated with the nearest residue and calculated features.
    """
    
    def __init__(self, x, y, z, normal_x=0, normal_y=0, normal_z=0):
        """
        Initialize a SAS point.
        
        Args:
            x, y, z (float): Coordinates of the SAS point
            normal_x, normal_y, normal_z (float): Components of the normal vector
        """
        self.x = x
        self.y = y
        self.z = z
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.normal_z = normal_z
        self.features = {}
        self.nearest_residue = None
        self.nearest_atom = None
        self.is_binding_site = False  # Could be set by comparing to known binding sites
        
    def get_coord(self):
        """Get the coordinates of the point as a numpy array."""
        return np.array([self.x, self.y, self.z])
    
    def get_normal(self):
        """Get the normal vector as a numpy array."""
        return np.array([self.normal_x, self.normal_y, self.normal_z])
    
    def distance(self, other):
        """
        Calculate the distance to another atom or point.
        
        Args:
            other: Another Atom, SASPoint, or coordinates as a numpy array
            
        Returns:
            float: Euclidean distance
        """
        if isinstance(other, (Atom, SASPoint)):
            other_coords = other.get_coord()
        else:  # Assume it's a numpy array or similar
            other_coords = other
        return np.linalg.norm(self.get_coord() - other_coords)
    
    def set_feature(self, name, value):
        """Set a feature value."""
        self.features[name] = value
    
    def get_feature(self, name, default=None):
        """Get a feature value with an optional default."""
        return self.features.get(name, default)
    
    def __str__(self):
        """String representation showing coordinates and nearest residue."""
        residue_info = "unknown"
        if self.nearest_residue:
            chain_id, residue_number = self.nearest_residue
            residue_info = f"chain {chain_id}, residue {residue_number}"
        
        return f"SASPoint(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, nearest_residue={residue_info})"


class Protein:
    """Class representing a protein structure with its atoms."""
    
    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        self.file_name = os.path.basename(pdb_path)
        self.atoms = []
        self.sas_points = []
        self.residues = {}  # Keyed by (chain_id, residue_number)
        self.ligands = []   # List of Ligand objects
        
        # Parse the PDB file
        self._parse_pdb()
    
    def _parse_pdb(self):
        """Parse the PDB file to extract atoms and ligands."""
        protein_atoms = []
        ligand_atoms_by_residue = {}  # Group ligand atoms by (chain_id, residue_number, residue_name)
        
        # Common solvent and metal ions that should not be considered as ligands
        excluded_ligands = {'HOH', 'WAT', 'H2O', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'SO4', 'PO4', 'ACE', 'NH2'}
        
        with open(self.pdb_path, 'r') as pdb_file:
            for line in pdb_file:
                # Process protein atoms (ATOM records)
                if line.startswith('ATOM'):
                    try:
                        atom_id = int(line[6:11].strip())
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21:22].strip()
                        residue_number = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        occupancy = float(line[54:60].strip() or 0.0)
                        temp_factor = float(line[60:66].strip() or 0.0)
                        element = line[76:78].strip().upper()
                        
                        # If element is empty, try to determine from atom name
                        if not element:
                            element = atom_name[0].upper()
                        
                        # Create the atom and add it to the list
                        atom = Atom(atom_id, atom_name, residue_name, chain_id, residue_number,
                                    x, y, z, occupancy, temp_factor, element)
                        protein_atoms.append(atom)
                        
                        # Group atoms by residue
                        residue_key = (chain_id, residue_number)
                        if residue_key not in self.residues:
                            self.residues[residue_key] = []
                        self.residues[residue_key].append(atom)
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing ATOM line in {self.pdb_path}: {e}")
                        logger.debug(f"Line content: {line}")
                
                # Process ligand atoms (HETATM records)
                elif line.startswith('HETATM'):
                    try:
                        atom_id = int(line[6:11].strip())
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21:22].strip()
                        residue_number = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        occupancy = float(line[54:60].strip() or 0.0)
                        temp_factor = float(line[60:66].strip() or 0.0)
                        element = line[76:78].strip().upper()
                        
                        # If element is empty, try to determine from atom name
                        if not element:
                            element = atom_name[0].upper()
                        
                        # Skip common solvent molecules and ions
                        if residue_name not in excluded_ligands:
                            # Create the atom
                            atom = Atom(atom_id, atom_name, residue_name, chain_id, residue_number,
                                       x, y, z, occupancy, temp_factor, element)
                            
                            # Group ligand atoms by residue
                            ligand_key = (chain_id, residue_number, residue_name)
                            if ligand_key not in ligand_atoms_by_residue:
                                ligand_atoms_by_residue[ligand_key] = []
                            ligand_atoms_by_residue[ligand_key].append(atom)
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing HETATM line in {self.pdb_path}: {e}")
                        logger.debug(f"Line content: {line}")
        
        # Create ligand objects from grouped atoms
        for ligand_key, atoms in ligand_atoms_by_residue.items():
            # Only consider ligands with at least 3 atoms (to filter out single ions)
            if len(atoms) >= 3:
                ligand = Ligand(atoms)
                self.ligands.append(ligand)
        
        # Store protein atoms
        self.atoms = protein_atoms
        
        logger.info(f"Parsed {len(self.atoms)} protein atoms and {len(self.ligands)} ligands from {self.file_name}")
        if self.ligands:
            ligand_info = [f"{l.name}({l.chain_id}:{l.residue_number})" for l in self.ligands]
            logger.info(f"Ligands found: {', '.join(ligand_info)}")
    
    def generate_sas_points(self, probe_radius=1.6, density=3.0):
        """
        Generate Solvent Accessible Surface points.
        
        This implementation uses a ray-casting approach to find surface points.
        For production use, consider FreeSASA, MSMS, or BioPython's DSSP/HSExposure.
        
        Args:
            probe_radius (float): Radius of the solvent probe (P2Rank default 1.6Å)
            density (float): Density of points (higher = more points, slower calculation)
        """
        logger.info(f"Generating SAS points for {self.file_name}")
        
        if not self.atoms:
            logger.warning(f"No atoms found in {self.file_name}")
            return
        
        # Get all coordinates and radii for faster processing
        coords = np.array([atom.get_coord() for atom in self.atoms])
        radii = np.array([ATOM_TYPES.get(atom.element, {'radius': 1.7})['radius'] for atom in self.atoms])
        
        # Generate initial points on spheres around each atom
        sas_points = []
        for i, atom in enumerate(self.atoms):
            # Skip hydrogen atoms and atoms with very small radii
            if atom.element == 'H' or radii[i] < 1.0:
                continue
                
            # Generate points on the extended atom sphere (atom radius + probe radius)
            radius = radii[i] + probe_radius
            n_points = max(10, int(density * radius * radius))  # Scale with surface area
            
            # Generate evenly distributed points on a sphere using the Fibonacci sphere algorithm
            points = []
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            for k in range(n_points):
                y = 1 - (k / float(n_points - 1)) * 2
                radius_at_y = radius * np.sqrt(1 - y * y)
                
                theta = 2 * np.pi * k / phi
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                
                point = np.array([x, y, z]) + atom.get_coord()
                points.append(point)
            
            # Check if each point is on the SAS (not inside any other atom + probe)
            for point in points:
                is_on_surface = True
                
                # Check against all other atoms
                for j, other_atom in enumerate(self.atoms):
                    if i == j:  # Skip the original atom
                        continue
                    
                    dist = np.linalg.norm(point - coords[j])
                    if dist < radii[j]:  # Point is inside another atom
                        is_on_surface = False
                        break
                
                if is_on_surface:
                    # Calculate normal vector (pointing outward from the atom)
                    normal = point - atom.get_coord()
                    normal = normal / np.linalg.norm(normal)
                    
                    # Create a SAS point
                    sas_point = SASPoint(point[0], point[1], point[2], normal[0], normal[1], normal[2])
                    sas_points.append(sas_point)
        
        # Subsample points if there are too many (for performance)
        #if len(sas_points) > 10000:
        #    logger.info(f"Subsampling SAS points from {len(sas_points)} to 10000")
        #    indices = np.linspace(0, len(sas_points)-1, 10000, dtype=int)
        #    sas_points = [sas_points[i] for i in indices]
        
        # Assign each SAS point to the nearest atom and residue
        for sas_point in sas_points:
            point_coord = sas_point.get_coord()

            nearest_atom = None
            min_atom_dist = float('inf')
            for atom in self.atoms:
                dist = atom.distance(point_coord)
                if dist < min_atom_dist:
                    min_atom_dist = dist
                    nearest_atom = atom

            sas_point.nearest_atom = nearest_atom
            if nearest_atom is not None:
                sas_point.nearest_residue = (nearest_atom.chain_id, nearest_atom.residue_number)
            else:
                sas_point.nearest_residue = None
        
        self.sas_points = sas_points
        logger.info(f"Generated {len(self.sas_points)} SAS points for {self.file_name}")
    
    def generate_biopython_sas_points(self, probe_radius=1.6, sr_n_points=100):
        """
        Generate Solvent Accessible Surface points using BioPython.
        
        This method uses BioPython's ShrakeRupley algorithm to calculate SASA values
        and then generates surface points where the SASA value is significant.
        
        Args:
            probe_radius (float): Radius of the solvent probe (P2Rank default 1.6Å)
            sr_n_points (int): Number of points to use in the ShrakeRupley algorithm
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not BIOPYTHON_AVAILABLE:
            logger.warning("BioPython is not available. Cannot generate surface points using BioPython.")
            return False
            
        logger.info(f"Generating SAS points for {self.file_name} using BioPython")
        
        try:
            # Parse the PDB file using BioPython
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", self.pdb_path)
            model = structure[0]  # Use the first model
            
            # Calculate SASA using ShrakeRupley algorithm
            sr = ShrakeRupley(probe_radius=probe_radius, n_points=sr_n_points)
            sr.compute(model, level="A")  # Atom level SASA
            
            # Generate SAS points
            sas_points = []
            
            # Create a neighbor search for the structure
            atoms = list(model.get_atoms())
            ns = NeighborSearch(atoms)
            
            # Process each atom with significant SASA
            for atom in atoms:
                # Skip hydrogen atoms
                if atom.element == 'H':
                    continue
                
                # Get the SASA value for this atom
                sasa = atom.sasa
                
                # Only consider atoms with significant SASA
                if sasa > 1.0:  # Threshold in square Angstroms
                    # Get atom properties
                    coord = atom.coord
                    element = atom.element
                    # Get atom radius from our constants (fallback to default if not found)
                    radius = ATOM_TYPES.get(element, {'radius': 1.7})['radius']
                    
                    # Scale points based on SASA and atom size
                    points_to_generate = max(1, int(sasa / (4 * np.pi * radius * radius) * 20))
                    points_to_generate = min(points_to_generate, 30)  # Cap at 30 points per atom
                    
                    # Use the Fibonacci sphere algorithm to generate evenly distributed points
                    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
                    
                    # Fix for division by zero when points_to_generate is 1
                    if points_to_generate <= 1:
                        # Just add one point directly above the atom
                        point = coord + np.array([0, 1, 0]) * (radius + probe_radius)
                        normal = np.array([0, 1, 0])
                        
                        # Create a SAS point
                        sas_point = SASPoint(point[0], point[1], point[2], normal[0], normal[1], normal[2])

                        residue = atom.get_parent()
                        chain = residue.get_parent()
                        chain_id = chain.id
                        residue_number = residue.id[1]

                        custom_atom = Atom(
                            atom.serial_number if hasattr(atom, 'serial_number') else -1,
                            atom.get_name(), residue.get_resname(), chain_id, residue_number,
                            coord[0], coord[1], coord[2],
                            float(atom.get_occupancy() or 0.0), float(atom.get_bfactor() or 0.0),
                            atom.element
                        )
                        sas_point.nearest_atom = custom_atom

                        # Get residue information
                        
                        # Set nearest residue
                        sas_point.nearest_residue = (chain_id, residue_number)
                        
                        sas_points.append(sas_point)
                    else:
                        for k in range(points_to_generate):
                            y = 1 - (k / float(points_to_generate - 1)) * 2
                            radius_at_y = radius * np.sqrt(1 - y * y)
                            
                            theta = 2 * np.pi * k / phi
                            x = np.cos(theta) * radius_at_y
                            z = np.sin(theta) * radius_at_y
                            
                            # Calculate point on the atom surface
                            point = coord + np.array([x, y, z]) * (radius + probe_radius)
                            
                            # Calculate normal vector pointing outward from the atom
                            normal = point - coord
                            normal = normal / np.linalg.norm(normal)
                            
                            # Create a SAS point
                            sas_point = SASPoint(point[0], point[1], point[2], normal[0], normal[1], normal[2])

                            residue = atom.get_parent()
                            chain = residue.get_parent()
                            chain_id = chain.id
                            residue_number = residue.id[1]

                            custom_atom = Atom(
                                atom.serial_number if hasattr(atom, 'serial_number') else -1,
                                atom.get_name(), residue.get_resname(), chain_id, residue_number,
                                coord[0], coord[1], coord[2],
                                float(atom.get_occupancy() or 0.0), float(atom.get_bfactor() or 0.0),
                                atom.element
                            )
                            sas_point.nearest_atom = custom_atom
                            
                            # Set nearest residue
                            sas_point.nearest_residue = (chain_id, residue_number)
                            
                            sas_points.append(sas_point)
            
            # Subsample points if there are too many (for performance)
            #if len(sas_points) > 10000:
            #    logger.info(f"Subsampling BioPython SAS points from {len(sas_points)} to 10000")
            #    indices = np.linspace(0, len(sas_points)-1, 10000, dtype=int)
            #    sas_points = [sas_points[i] for i in indices]
            
            # If we don't have enough points, we might need to lower the SASA threshold
            if len(sas_points) < 1000 and len(atoms) > 0:
                logger.warning(f"Generated only {len(sas_points)} points, retrying with lower SASA threshold")
                
                # Clear points and retry with lower threshold
                sas_points = []
                
                # Use a lower threshold for SASA
                for atom in atoms:
                    if atom.element == 'H':
                        continue
                    
                    sasa = atom.sasa
                    
                    # Lower threshold
                    if sasa > 0.5:  # More permissive threshold
                        coord = atom.coord
                        element = atom.element
                        radius = ATOM_TYPES.get(element, {'radius': 1.7})['radius']
                        
                        # Generate at least one point per exposed atom
                        points_to_generate = max(1, int(sasa / 20) + 1)
                        points_to_generate = min(points_to_generate, 10)
                        
                        for k in range(points_to_generate):
                            # Similar point generation logic as above
                            y = 1 - (k / float(points_to_generate - 1)) * 2 if points_to_generate > 1 else 0
                            radius_at_y = radius * np.sqrt(1 - y * y)
                            
                            theta = 2 * np.pi * k / (points_to_generate) if points_to_generate > 1 else 0
                            x = np.cos(theta) * radius_at_y
                            z = np.sin(theta) * radius_at_y
                            
                            point = coord + np.array([x, y, z]) * (radius + probe_radius)
                            normal = point - coord
                            normal = normal / np.linalg.norm(normal)
                            
                            sas_point = SASPoint(point[0], point[1], point[2], normal[0], normal[1], normal[2])
                            
                            # Get residue information directly from the atom
                            residue = atom.get_parent()
                            chain = residue.get_parent()
                            chain_id = chain.id
                            residue_number = residue.id[1]
                            
                            sas_point.nearest_residue = (chain_id, residue_number)
                            sas_points.append(sas_point)
            
            # Store the SAS points
            self.sas_points = sas_points
            logger.info(f"Generated {len(self.sas_points)} SAS points for {self.file_name} using BioPython")
            return True
            
        except Exception as e:
            logger.error(f"Error generating BioPython SAS points for {self.file_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def calculate_features(self, neighborhood_radius=6.0, protrusion_radius=11.3, atom_table_feat_pow=2.0, atom_table_feat_keep_sgn=False):
        """Calculate P2Rank-like features for each SAS point using neighbourhood aggregation."""
        logger.info(
            f"Calculating features for {len(self.sas_points)} SAS points in {self.file_name}")

        def weight_inv(dist):
            w = 1.0 - dist / neighborhood_radius
            return w if w > 0 else 0.0

        # pre-compute atom feature vectors
        atom_vectors = {}
        for atom in self.atoms:
            vec = {}
            chem_vec = surface_chem_vector(atom)
            for h in CHEM_HEADER:
                vec[f'chem.{h}'] = chem_vec.get(h, 0.0)

            key = f"{atom.residue_name}.{atom.atom_name}"
            vs = volsite_atom_properties(atom.atom_name, atom.residue_name)
            for h in VOLSITE_HEADER:
                vec[f'volsite.{h}'] = float(vs.get(h, 0.0))

            at = ATOMIC_TABLE.get(key)
            if at is None:
                at = {h: 0.0 for h in ATOM_TABLE_HEADER}
            for h in ATOM_TABLE_HEADER:
                val = float(at.get(h, 0.0))
                if atom_table_feat_keep_sgn:
                    sign = 1.0 if val >= 0 else -1.0
                    vec[f'atom_table.{h}'] = sign * (abs(val) ** atom_table_feat_pow)
                else:
                    vec[f'atom_table.{h}'] = abs(val) ** atom_table_feat_pow

            vec['bfactor.bfactor'] = atom.temp_factor

            atom_vectors[atom] = vec

        for sas_point in self.sas_points:
            point_coord = sas_point.get_coord()

            neighbours = []
            min_d = float('inf')
            nearest_atom = None
            for atom in self.atoms:
                dist = atom.distance(point_coord)
                if dist < min_d:
                    min_d = dist
                    nearest_atom = atom
                if dist <= neighborhood_radius:
                    neighbours.append((atom, dist))

            features = {
                'file_name': self.file_name,
                'x': sas_point.x,
                'y': sas_point.y,
                'z': sas_point.z,
                'chain_id': nearest_atom.chain_id if nearest_atom else 'UNK',
                'residue_number': nearest_atom.residue_number if nearest_atom else -1,
                'residue_name': nearest_atom.residue_name if nearest_atom else 'UNK',
                'class': 0
            }

            agg = {f'chem.{h}': 0.0 for h in CHEM_HEADER}
            for h in VOLSITE_HEADER:
                agg[f'volsite.{h}'] = 0.0
            for h in ATOM_TABLE_HEADER:
                agg[f'atom_table.{h}'] = 0.0
            agg['bfactor.bfactor'] = 0.0

            for atom, dist in neighbours:
                weight = weight_inv(dist)
                vec = atom_vectors[atom]
                for k, v in vec.items():
                    agg[k] += v * weight

            prot_atoms = sum(1 for a in self.atoms if a.distance(point_coord) <= protrusion_radius)
            agg['protrusion.protrusion'] = float(prot_atoms)
            agg['chem.atoms'] = float(len(neighbours))

            for k, v in agg.items():
                features[k] = v

            sas_point.features = features
    
    def export_features(self, output_path):
        """Export features to a CSV file."""
        if not self.sas_points:
            logger.warning(f"No SAS points to export for {self.file_name}")
            return
        
        # Extract features from SAS points
        features_dicts = []
        for point in self.sas_points:
            features = point.features.copy()
            # Set the class based on binding site classification
            features['class'] = 1 if point.is_binding_site else 0
            features_dicts.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_dicts)

        feature_cols = [f'chem.{h}' for h in CHEM_HEADER] + \
                       [f'volsite.{h}' for h in VOLSITE_HEADER] + \
                       ['protrusion.protrusion', 'bfactor.bfactor'] + \
                       [f'atom_table.{h}' for h in ATOM_TABLE_HEADER]
        meta_cols = ['file_name', 'x', 'y', 'z', 'chain_id', 'residue_number', 'residue_name']
        order = meta_cols + feature_cols + ['class']
        df = df[order]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} feature rows to {output_path}")
    
    def classify_binding_sites(self, binding_site_distance_threshold=4.0):
        """
        Classify SAS points as binding sites based on their distance to ligands.
        
        Args:
            binding_site_distance_threshold (float): Maximum distance (in Angstroms) 
                                                  from a SAS point to a ligand atom
                                                  to be classified as a binding site
        """
        logger.info(f"Classifying binding sites for {self.file_name} with threshold {binding_site_distance_threshold}Å")
        
        if not self.ligands:
            logger.warning(f"No ligands found in {self.file_name}, no binding sites will be classified")
            # Set all points as non-binding sites
            for sas_point in self.sas_points:
                sas_point.is_binding_site = False
            return
        
        logger.info(f"Found {len(self.ligands)} ligands in {self.file_name}")
        binding_site_count = 0
        
        for sas_point in self.sas_points:
            # Calculate minimum distance to any ligand atom
            min_distance = float('inf')
            for ligand in self.ligands:
                distance = ligand.closest_distance_to_point(sas_point)
                min_distance = min(min_distance, distance)
            
            # Classify as binding site if within the threshold
            if min_distance <= binding_site_distance_threshold:
                sas_point.is_binding_site = True
                binding_site_count += 1
            else:
                sas_point.is_binding_site = False
        
        binding_site_percentage = (binding_site_count / len(self.sas_points) * 100) if self.sas_points else 0
        logger.info(f"Classified {binding_site_count} out of {len(self.sas_points)} SAS points "
                   f"({binding_site_percentage:.2f}%) as binding sites in {self.file_name}")


class Ligand:
    """Class representing a ligand molecule in a protein structure."""
    
    def __init__(self, atoms):
        """
        Initialize a ligand with its atoms.
        
        Args:
            atoms (list): List of Atom objects belonging to this ligand
        """
        self.atoms = atoms
        self.name = atoms[0].residue_name if atoms else "UNK"
        self.chain_id = atoms[0].chain_id if atoms else "X"
        self.residue_number = atoms[0].residue_number if atoms else 0
        
    def get_coords(self):
        """Get the coordinates of all atoms in the ligand."""
        return np.array([atom.get_coord() for atom in self.atoms])
    
    def get_center(self):
        """Get the center coordinates of the ligand."""
        coords = self.get_coords()
        return np.mean(coords, axis=0) if len(coords) > 0 else np.array([0, 0, 0])
    
    def closest_distance_to_point(self, point):
        """
        Calculate the shortest distance from any ligand atom to the given point.
        
        Args:
            point: Point coordinates as numpy array or an object with get_coord() method
            
        Returns:
            float: Minimum distance from the point to any ligand atom
        """
        if isinstance(point, SASPoint) or isinstance(point, Atom):
            point_coord = point.get_coord()
        else:
            point_coord = point
            
        distances = [atom.distance(point_coord) for atom in self.atoms]
        return min(distances) if distances else float('inf')


def parse_dataset_file(dataset_path):
    """Parse a dataset file to get PDB file paths."""
    pdb_paths = []
    base_dir = os.path.dirname(dataset_path)
    
    with open(dataset_path, 'r') as ds_file:
        for line in ds_file:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('PARAM.'):
                continue
            
            # Construct the full path to the PDB file
            pdb_path = os.path.join(base_dir, line)
            
            # Check if file exists
            if os.path.exists(pdb_path):
                pdb_paths.append(pdb_path)
            else:
                logger.warning(f"PDB file not found: {pdb_path}")
    
    return pdb_paths


def is_feature_file_valid(file_path, required_cols=None):
    """Check if a feature CSV file is valid and not corrupted.

    The function attempts to read a small portion of the file and verify that
    essential columns are present. If the file cannot be read or required
    columns are missing, it is considered invalid.

    Args:
        file_path (str): Path to the feature file.
        required_cols (list, optional): Columns that must be present in the
            file. Defaults to a minimal set of metadata columns.

    Returns:
        bool: True if the file appears valid, False otherwise.
    """

    if required_cols is None:
        required_cols = [
            "file_name",
            "x",
            "y",
            "z",
            "chain_id",
            "residue_number",
            "residue_name",
            "class",
        ]

    try:
        if os.path.getsize(file_path) == 0:
            logger.warning(f"Feature file {file_path} is empty")
            return False

        df = pd.read_csv(file_path, nrows=1)
        for col in required_cols:
            if col not in df.columns:
                logger.warning(
                    f"Feature file {file_path} is missing required column '{col}'"
                )
                return False

        return True
    except Exception as e:
        logger.warning(f"Failed to read feature file {file_path}: {e}")
        return False


def process_protein(pdb_path, output_dir):
    """Process a single protein and extract features."""
    try:
        # Create output path
        protein_name = os.path.basename(pdb_path)
        output_path = os.path.join(output_dir, f"{protein_name}_features.csv")

        # Skip if output already exists
        if os.path.exists(output_path):
            logger.info(f"Output already exists for {protein_name}, skipping")
            return

        # Process the protein
        protein = Protein(pdb_path)

        # Prioritize BioPython for SAS point generation
        if BIOPYTHON_AVAILABLE:
            logger.info(f"Attempting SAS point generation with BioPython for {protein_name}.")
            success_biopython = protein.generate_biopython_sas_points()
            if success_biopython and protein.sas_points:
                logger.info(f"Successfully generated {len(protein.sas_points)} SAS points using BioPython for {protein_name}.")
            else:
                if not success_biopython:
                    logger.warning(f"BioPython SAS point generation failed for {protein_name}.")
                else:  # success_biopython is True but no points
                    logger.warning(f"BioPython SAS point generation succeeded but resulted in 0 points for {protein_name}.")
                logger.info(f"Falling back to custom SAS point generation for {protein_name}.")
                protein.generate_sas_points()  # Fallback to custom method
                if protein.sas_points:
                    logger.info(f"Successfully generated {len(protein.sas_points)} SAS points using custom method for {protein_name}.")
                else:
                    logger.warning(f"Custom SAS point generation also resulted in 0 points for {protein_name}.")
        else:
            logger.info(f"BioPython not available. Using custom SAS point generation for {protein_name}.")
            protein.generate_sas_points()  # Use custom if BioPython is not there at all
            if protein.sas_points:
                logger.info(f"Successfully generated {len(protein.sas_points)} SAS points using custom method for {protein_name}.")
            else:
                logger.warning(f"Custom SAS point generation resulted in 0 points for {protein_name} (BioPython not available).")

        # Continue only if SAS points were generated
        if not protein.sas_points:
            logger.error(f"Failed to generate SAS points for {protein_name} using any available method. Skipping feature extraction.")
            return None

        protein.calculate_features(
            neighborhood_radius=neighborhood_radius,
            protrusion_radius=protrusion_radius,
            atom_table_feat_pow=atom_table_feat_pow
        )
        # Classify binding sites based on ligand proximity
        protein.classify_binding_sites()
        protein.export_features(output_path)

        return output_path
    except Exception as e:
        logger.error(f"Error processing protein {pdb_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def combine_features(output_dir):
    """Combine all feature CSV files into a single file."""
    feature_files = glob.glob(os.path.join(output_dir, "*_features.csv"))
    
    if not feature_files:
        logger.warning(f"No feature files found in {output_dir}")
        return
    
    # Read and combine all CSVs
    dfs = []
    for file_path in feature_files:
        if not is_feature_file_valid(file_path):
            logger.warning(f"Skipping invalid feature file: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    if not dfs:
        logger.warning("No valid feature files could be read")
        return
    
    # Combine all features and write to a single file
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_output_path = os.path.join(output_dir, "vectorsTrain.csv")
    combined_df.to_csv(combined_output_path, index=False)
    
    logger.info(f"Combined {len(combined_df)} rows into {combined_output_path}")
    return combined_output_path


def process_with_params(pdb_path, output_dir, use_biopython, probe_radius, sr_n_points, point_density, neighborhood_radius, protrusion_radius, atom_table_feat_pow):
    """Process a single protein and extract features with the given parameters."""
    try:
        protein_name = os.path.basename(pdb_path)
        output_path = os.path.join(output_dir, f"{protein_name}_features.csv")
        
        # Process the protein
        start_time = time.time()
        protein = Protein(pdb_path)
        
        # Generate SAS points using either BioPython or custom implementation
        if use_biopython and BIOPYTHON_AVAILABLE:
            success = protein.generate_biopython_sas_points(
                probe_radius=probe_radius, 
                sr_n_points=sr_n_points
            )
            if not success:
                logger.warning(f"Failed to generate SAS points using BioPython for {protein_name}. Falling back to custom implementation.")
                protein.generate_sas_points(probe_radius=probe_radius, density=point_density)
        else:
            protein.generate_sas_points(probe_radius=probe_radius, density=point_density)
        
        protein.calculate_features(
            neighborhood_radius=neighborhood_radius,
            protrusion_radius=protrusion_radius,
            atom_table_feat_pow=atom_table_feat_pow
        )
        # Classify binding sites based on ligand proximity
        protein.classify_binding_sites()
        protein.export_features(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processed {protein_name} in {elapsed_time:.2f} seconds")
        
        return output_path
    except Exception as e:
        logger.error(f"Error processing protein {pdb_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main function to run the feature extraction pipeline."""
    
    parser = argparse.ArgumentParser(description="Extract protein features from PDB files for machine learning")
    # Required arguments
    parser.add_argument("dataset_file", help="Path to the dataset (.ds) file containing PDB file paths")
    parser.add_argument("output_dir", help="Directory to save the output files")
    
    # Optional arguments
    parser.add_argument("--probe_radius", type=float, default=1.6, help="Solvent probe radius in Angstroms (default: 1.6)")
    parser.add_argument("--point_density", type=float, default=3.0, help="SAS point density approximating P2Rank tessellation=2")
    parser.add_argument("--neighborhood_radius", type=float, default=6.0, help="Radius for feature calculation (default: 6.0)")
    parser.add_argument("--protrusion_radius", type=float, default=11.3, help="Radius for protrusion calculation (default: 11.3)")
    parser.add_argument("--atom_table_feat_pow", type=float, default=2.0, help="Power for atom table features (default: 2.0)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help=f"Number of processing threads (default: {os.cpu_count()})")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing proteins that already have feature files")
    parser.add_argument("--validate", action="store_true", help="Validate the final output file")
    parser.add_argument("--analyze", action="store_true", help="Analyze the output and generate statistics")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze existing output without processing proteins")
    parser.add_argument("--use_biopython", action="store_true", help="Use BioPython for SAS point generation instead of the custom implementation")
    parser.add_argument("--sr_n_points", type=int, default=100, help="Number of points to use in the ShrakeRupley algorithm (default: 100)")
    parser.add_argument("--atom_table_feat_keep_sgn", action="store_true", default=False, help="Preserve sign when applying exponent to atom table features (default: False, matches P2Rank default)")

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If analysis only mode is enabled, just analyze the existing output
    if args.analyze_only:
        output_file = os.path.join(args.output_dir, "vectorsTrain.csv")
        if os.path.exists(output_file):
            analyze_feature_file(output_file, os.path.join(args.output_dir, "analysis"))
            return
        else:
            logger.error(f"Output file {output_file} not found for analysis. Exiting.")
            sys.exit(1)
    
    # Normal processing mode
    logger.info(f"Processing dataset: {args.dataset_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(
        f"Parameters: probe_radius={args.probe_radius}, point_density={args.point_density}, "
        f"neighborhood_radius={args.neighborhood_radius}, protrusion_radius={args.protrusion_radius}, "
        f"atom_table_feat_pow={args.atom_table_feat_pow}"
    )
    
    # Get the number of threads
    num_threads = args.threads
    logger.info(f"Using {num_threads} threads for processing")
    
    if args.use_biopython:
        if BIOPYTHON_AVAILABLE:
            logger.info("Using BioPython for SAS point generation")
        else:
            logger.warning("BioPython requested but not available. Falling back to custom implementation.")
    else:
        logger.info("Using custom ray-casting implementation for SAS point generation")
    
    # Parse the dataset file
    pdb_paths = parse_dataset_file(args.dataset_file)
    logger.info(f"Found {len(pdb_paths)} PDB files in the dataset")
    
    if not pdb_paths:
        logger.error("No valid PDB files found. Exiting.")
        sys.exit(1)
    
    # Filter out proteins that have already been processed if skip_existing is True
    if args.skip_existing:
        filtered_paths = []
        for pdb_path in pdb_paths:
            protein_name = os.path.basename(pdb_path)
            output_path = os.path.join(args.output_dir, f"{protein_name}_features.csv")
            if os.path.exists(output_path):
                if is_feature_file_valid(output_path):
                    logger.info(f"Skipping already processed protein: {protein_name}")
                    continue
                else:
                    logger.warning(
                        f"Existing feature file for {protein_name} is invalid. Re-processing."
                    )
            filtered_paths.append(pdb_path)

        logger.info(
            f"Processing {len(filtered_paths)} out of {len(pdb_paths)} proteins"
        )
        pdb_paths = filtered_paths
    
    # Process proteins in parallel
    output_paths = []
    
    if num_threads > 1 and pdb_paths:
        # Use a thread pool for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            # Create a list of tuples with parameters for each protein
            process_params = [
                (
                    pdb_path,
                    args.output_dir,
                    args.use_biopython,
                    args.probe_radius,
                    args.sr_n_points,
                    args.point_density,
                    args.neighborhood_radius,
                    args.protrusion_radius,
                    args.atom_table_feat_pow,
                )
                for pdb_path in pdb_paths
            ]
            
            # Map the process_with_params function to the parameter list
            futures = [executor.submit(process_with_params, *params) for params in process_params]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    output_path = future.result()
                    if output_path:
                        output_paths.append(output_path)
                        logger.info(f"Successfully processed a protein")
                    else:
                        logger.warning(f"Failed to process a protein")
                except Exception as e:
                    logger.error(f"Error processing a protein: {e}")
    else:
        # Process sequentially for single thread or empty list
        for pdb_path in pdb_paths:
            output_path = process_with_params(
                pdb_path,
                args.output_dir,
                args.use_biopython,
                args.probe_radius,
                args.sr_n_points,
                args.point_density,
                args.neighborhood_radius,
                args.protrusion_radius,
                args.atom_table_feat_pow,
            )
            if output_path:
                output_paths.append(output_path)
    
    # Combine all feature files
    combined_path = combine_features(args.output_dir)
    
    if combined_path:
        logger.info(f"Feature extraction complete. Combined features saved to {combined_path}")
        
        # Validate the output if requested
        if args.validate and os.path.exists(combined_path):
            try:
                validate_output(combined_path)
            except Exception as e:
                logger.error(f"Validation failed: {e}")
        
        # Analyze the output if requested
        if args.analyze and os.path.exists(combined_path):
            try:
                analyze_feature_file(combined_path, os.path.join(args.output_dir, "analysis"))
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
    else:
        logger.error("Feature extraction failed. No combined output was generated.")


def validate_output(output_path):
    """Validate the output CSV file."""
    logger.info(f"Validating output file: {output_path}")
    
    try:
        df = pd.read_csv(output_path)
        
        # Check if the file is empty
        if df.empty:
            logger.error("Output file is empty")
            return False
        
        # Check if all required columns are present
        required_columns = ['file_name', 'x', 'y', 'z', 'chain_id', 'residue_number']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check if there are any NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found {nan_counts.sum()} NaN values in the output")
            logger.warning(f"NaN counts by column: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check the range of values for key numerical features
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].min() < -1e10 or df[col].max() > 1e10:
                logger.warning(f"Column {col} has extreme values: min={df[col].min()}, max={df[col].max()}")
        
        # Count unique proteins
        protein_count = df['file_name'].nunique()
        logger.info(f"Output contains data for {protein_count} unique proteins")
        logger.info(f"Total number of SAS points: {len(df)}")
        
        # Display summary of residue chains
        chain_counts = df['chain_id'].value_counts()
        logger.info(f"Chain ID distribution: {dict(chain_counts)}")
        
        logger.info("Validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating output: {e}")
        import traceback



        logger.error(traceback.format_exc())
        return False


def analyze_feature_file(feature_file, output_dir=None):
    """
    Analyze a feature file and generate statistics and visualizations.
    
    Args:
        feature_file (str): Path to the feature file (CSV)
        output_dir (str, optional): Directory to save analysis outputs
    
    Returns:
        dict: Dictionary with analysis results
    """
    logger.info(f"Analyzing feature file: {feature_file}")
    
    try:
        # Read the feature file
        df = pd.read_csv(feature_file)
        
        if df.empty:
            logger.warning("Feature file is empty")
            return {}
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Basic statistics
        stats = {
            'total_points': len(df),
            'proteins': df['file_name'].nunique(),
            'chains': df['chain_id'].nunique(),
            'residues': df.groupby(['chain_id', 'residue_number']).ngroups
        }
        
        # Feature statistics
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        feature_stats = df[numeric_cols].describe().to_dict()
        stats['feature_stats'] = feature_stats
        
        # Chain and residue distribution
        chain_counts = df['chain_id'].value_counts().to_dict()
        residue_counts = df['residue_name'].value_counts().to_dict()
        stats['chain_counts'] = chain_counts
        stats['residue_counts'] = residue_counts
        
        # Print basic statistics
        logger.info(f"Total SAS points: {stats['total_points']}")
        logger.info(f"Unique proteins: {stats['proteins']}")
        logger.info(f"Unique chains: {stats['chains']}")
        logger.info(f"Unique residues: {stats['residues']}")
        logger.info(f"Chain distribution: {chain_counts}")
        logger.info(f"Top 5 residue types: {dict(sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        # Generate detailed report if output directory is provided
        if output_dir:
            # Save summary statistics as CSV
            summary_df = pd.DataFrame({
                'Metric': ['Total Points', 'Unique Proteins', 'Unique Chains', 'Unique Residues'],
                'Value': [stats['total_points'], stats['proteins'], stats['chains'], stats['residues']]
            })
            summary_df.to_csv(os.path.join(output_dir, 'summary_stats.csv'), index=False)
            
            # Save feature distributions as CSV
            feature_dist_df = df[numeric_cols].describe().T
            feature_dist_df.to_csv(os.path.join(output_dir, 'feature_stats.csv'))
            
            # Save residue and chain counts
            pd.DataFrame(list(chain_counts.items()), columns=['Chain', 'Count']).to_csv(
                os.path.join(output_dir, 'chain_counts.csv'), index=False)
            pd.DataFrame(list(residue_counts.items()), columns=['Residue', 'Count']).to_csv(
                os.path.join(output_dir, 'residue_counts.csv'), index=False)
            
            logger.info(f"Analysis reports saved to {output_dir}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Error analyzing feature file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


if __name__ == "__main__":
    main()
