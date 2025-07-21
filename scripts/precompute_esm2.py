import os
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
import esm
from glob import glob

import os
# redirect torch/esm caches to your big data path
os.environ["TORCH_HOME"]        = "/system/user/studentwork/hageneder/cache/torch"
os.environ["TORCH_MODEL_ZOO"]   = "/system/user/studentwork/hageneder/cache/torch"
os.environ["TRANSFORMERS_CACHE"] = "/system/user/studentwork/hageneder/cache/transformers"
os.environ["HF_HOME"]            = "/system/user/studentwork/hageneder/cache/hf"
# 1) Constants
PDB_DIR = "/system/user/studentwork/hageneder/MSC/Practical_work/p2rank-datasets/chen11"
OUT_DIR = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/esm2"
DEVICE = "cuda:3"

# 2) Load ESM-2
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
model = model.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()

# 3) PDB → sequence
p = PDBParser(QUIET=True)
ppb = PPBuilder()

def pdb_to_seq(pdb_path):
    # Take chain A (or first) only
    structure = p.get_structure("", pdb_path)
    for model in structure:
        for chain in model:
            # build peptide → list of SeqRecords
            peptides = ppb.build_peptides(chain)
            if peptides:
                return str(peptides[0].get_sequence())
    return None

# 4) Iterate
for pdb_file in glob(os.path.join(PDB_DIR, "*.pdb")):
    pid = os.path.splitext(os.path.basename(pdb_file))[0]
    seq = pdb_to_seq(pdb_file)
    if not seq:
        print(f"{pid}: no chain found, skipping")
        continue

    # format for ESM batch converter
    batch = [ (pid, seq) ]
    labels, strs, toks = batch_converter(batch)
    toks = toks.to(DEVICE)

    # Extract representations
    with torch.no_grad():
        out = model(toks, repr_layers=[33], return_contacts=False)
    # shape [batch, seq_len, dim]
    reps = out["representations"][33].squeeze(0).cpu()

    # Save
    torch.save(reps, os.path.join(OUT_DIR, f"{pid}.pt"))
    print(f"Wrote {pid}.pt  →  {reps.shape}")
