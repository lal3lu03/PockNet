#!/usr/bin/env python3
"""
Regenerate chain-level ESM2-3B embeddings with the canonical naming scheme.

For every PDB listed in one or more .ds files, this script will:
  • parse all standard amino-acid residues (per chain, residue-level)
  • encode them using esm2_t36_3B_UR50D
  • save torch files of the form <protein_id>_<CHAIN>.pt
      {
          "emb":     FloatTensor [num_residues, 2560],
          "resnums": List[int]   residue numbers (PDB numbering)
      }
  • skip existing embeddings unless --force is supplied

Key features:
  • Uses the shared cache: /system/user/studentwork/hageneder/cache
  • Accepts multiple --ds-file arguments
  • Can restrict to a list of bases (e.g., missing_esm_pdb_ids.txt)
  • Supports multi-GPU via --split-index / --split-count
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, Set

import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
import esm


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_base_and_hint(stem: str) -> Tuple[str, Optional[str]]:
    """
    Normalize a PDB stem to the base identifier used in CSV & embeddings.

    Examples:
        1krn_A -> base=1krn,  chain_hint=A
        1a26A  -> base=1a26, chain_hint=A
        a.001.001.001_1s69a -> base=a.001.001.001_1s69a, chain_hint=None
    """
    base = stem.lower()
    chain_hint = None

    if "_" in base:
        left, right = base.rsplit("_", 1)
        if len(right) <= 2 and right.isalpha():
            base = left
            chain_hint = right.upper()
    else:
        orig = stem
        if orig and orig[-1].isalpha() and orig[-1].isupper():
            base = base[:-1]
            chain_hint = orig[-1]

    return base, chain_hint


def collect_pdbs(ds_paths: Iterable[Path],
                 pdb_base: Path,
                 only_set: Optional[Set[str]]) -> list[Tuple[Path, str]]:
    """
    Collect PDB paths and corresponding base names from .ds files.

    Returns list of (absolute_path, base_name).
    """
    entries = []
    for ds_path in ds_paths:
        if not ds_path.exists():
            raise FileNotFoundError(f".ds file not found: {ds_path}")
        for raw_line in ds_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("PARAM."):
                continue
            rel_path = Path(line)
            abs_path = pdb_base / rel_path
            stem = rel_path.stem
            base, _ = normalize_base_and_hint(stem)
            if only_set and base not in only_set:
                continue
            entries.append((abs_path, base))
    return entries


def extract_residues(chain):
    """
    Yield (one_letter_code, seq_id) for standard amino-acid residues that have a CA atom.
    """
    for residue in chain:
        hetero_flag = residue.id[0]
        if hetero_flag != " ":
            continue  # skip heteroatoms, waters, ligands
        if not is_aa(residue, standard=True):
            continue
        if "CA" not in residue:
            continue
        try:
            aa = three_to_one(residue.get_resname())
        except KeyError:
            continue
        seq_id = residue.id[1]
        yield aa, seq_id


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(entries: list[Tuple[Path, str]],
                        out_dir: Path,
                        device: torch.device,
                        only_set: Optional[Set[str]],
                        force: bool,
                        split_index: int,
                        split_count: int):
    """
    Generate ESM2 embeddings for the supplied PDB entries.
    """
    # Set shared cache locations
    os.environ["TORCH_HOME"] = "/system/user/studentwork/hageneder/cache/torch"
    os.environ["TORCH_MODEL_ZOO"] = "/system/user/studentwork/hageneder/cache/torch"
    os.environ["TRANSFORMERS_CACHE"] = "/system/user/studentwork/hageneder/cache/transformers"
    os.environ["HF_HOME"] = "/system/user/studentwork/hageneder/cache/hf"

    print("=" * 80)
    print("Generating ESM2 embeddings (esm2_t36_3B_UR50D)")
    print("=" * 80)
    print(f"Target directory : {out_dir}")
    print(f"Device           : {device}")
    print(f"Total entries    : {len(entries)}")
    print(f"Split index      : {split_index} / {split_count}")
    print(f"Force overwrite  : {force}")
    print("=" * 80)

    out_dir.mkdir(parents=True, exist_ok=True)

    esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    esm_model = esm_model.to(device).eval()
    # Determine number of layers for repr_layers
    num_layers = getattr(esm_model, "num_layers", None)
    if num_layers is None:
        # Some esm versions expose the underlying model via .model
        inner = getattr(esm_model, "model", None)
        if inner is not None:
            num_layers = getattr(inner, "num_layers", None)
    if num_layers is None:
        # Fall back to length of encoder layers if available
        encoder = getattr(esm_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            num_layers = len(encoder.layers)
    if num_layers is None:
        raise AttributeError("Could not determine number of layers for ESM model")

    batch_converter = alphabet.get_batch_converter()
    parser = PDBParser(QUIET=True)

    processed_keys: Set[str] = set()
    saved, skipped, missing = 0, 0, 0

    for idx, (pdb_path, base) in enumerate(entries):
        if idx % split_count != split_index:
            continue  # Assigned to another worker

        if not pdb_path.exists():
            print(f"[missing] {pdb_path}")
            missing += 1
            continue

        try:
            structure = parser.get_structure(base, pdb_path)
        except Exception as exc:
            print(f"[error] Failed to parse {pdb_path}: {exc}")
            continue

        for model in structure:
            for chain in model:
                raw_id = chain.id
                alias = raw_id.strip() or "A"
                chain_id_upper = alias.upper()
                key = f"{base}_{chain_id_upper}"
                if key in processed_keys and not force:
                    continue

                out_file = out_dir / f"{key}.pt"
                if out_file.exists() and not force:
                    skipped += 1
                    processed_keys.add(key)
                    continue

                residues = list(extract_residues(chain))
                if not residues:
                    continue

                letters, resnums = zip(*residues)
                sequence = "".join(letters)
                if not sequence:
                    continue

                batch = [(key, sequence)]
                _, _, tokens = batch_converter(batch)
                tokens = tokens.to(device)

                with torch.no_grad():
                    result = esm_model(tokens, repr_layers=[num_layers], return_contacts=False)
                token_reps = result["representations"][num_layers]
                emb = token_reps[0, 1 : len(sequence) + 1].cpu().float()

                # Save with complete metadata
                torch.save({
                    "emb": emb,
                    "resnums": list(resnums),
                    "protein_id": base,
                    "chain_id": chain_id_upper,
                    "model_name": "esm2_t36_3B_UR50D"
                }, out_file)
                processed_keys.add(key)
                saved += 1

                if saved % 50 == 0:
                    print(f"[progress] {saved} saved, {skipped} skipped, {missing} missing")

    print("=" * 80)
    print("Generation complete")
    print(f"Saved   : {saved} (new embeddings generated)")
    print(f"Skipped : {skipped} (already present)")
    print(f"Missing : {missing} (PDB files not found)")
    print(f"Total   : {saved + skipped} embeddings available")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate chain-level ESM2-3B embeddings with canonical naming."
    )
    parser.add_argument(
        "--ds-file",
        action="append",
        required=True,
        help="Path to a .ds file (you can pass multiple --ds-file flags)."
    )
    parser.add_argument(
        "--pdb-base",
        type=Path,
        default=Path("data/p2rank-datasets"),
        help="Base directory containing PDB datasets."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/esm2_3B_chain"),
        help="Target directory for embeddings."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (e.g., cuda:1). Falls back to CPU if unavailable."
    )
    parser.add_argument(
        "--only",
        type=Path,
        help="Optional text file listing base IDs to process (one per line)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embeddings even if the .pt file already exists."
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Worker index for multi-GPU runs (0-based)."
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=1,
        help="Total number of parallel splits."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    only_set = None
    if args.only:
        if not args.only.exists():
            raise FileNotFoundError(args.only)
        only_set = {line.strip().lower() for line in args.only.read_text().splitlines() if line.strip()}

    ds_paths = [Path(p) for p in args.ds_file]
    entries = collect_pdbs(ds_paths, args.pdb_base, only_set)

    generate_embeddings(
        entries=entries,
        out_dir=args.out_dir,
        device=device,
        only_set=only_set,
        force=args.force,
        split_index=args.split_index,
        split_count=args.split_count,
    )


if __name__ == "__main__":
    main()
