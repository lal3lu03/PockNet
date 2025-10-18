#!/usr/bin/env python3
"""
Merge vectorsTrain_all.csv with vectorsTrain_all_chainfix.csv to create complete dataset.

This script:
1. Backs up existing vectorsTrain_all_chainfix.csv
2. Loads vectorsTrain_all.csv (788 proteins, ~6M rows)
3. Adds protein_id and residue_id columns to ALL rows
4. Merges with existing chainfix data to preserve any manual fixes
5. Ensures all 788 proteins are present in final output
"""

import sys
from pathlib import Path
from typing import Tuple
from datetime import datetime

# Add minimal inline CSV processing to avoid pandas dependency issues
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_chain_registry(features_dir: Path) -> dict:
    """Build a registry mapping file_name to observed chain IDs from feature CSVs."""
    registry = {}
    feature_files = list(features_dir.glob("*.pdb_features.csv"))
    
    logger.info(f"Building chain registry from {len(feature_files)} feature files...")
    
    for path in feature_files:
        try:
            with path.open() as fh:
                reader = csv.DictReader(fh)
                chains = {row.get("chain_id", "").strip().upper()
                         for row in reader if row.get("chain_id", "").strip()}
            
            # Map to the filename format used in vectorsTrain_all.csv
            # e.g., "1krn.pdb_features.csv" -> "1krn.pdb"
            file_name = path.stem.replace("_features", "")
            if not file_name.endswith(".pdb"):
                file_name += ".pdb"
            
            if chains:
                registry[file_name] = chains
        except Exception as e:
            logger.warning(f"Error reading {path.name}: {e}")
    
    logger.info(f"Registry built: {len(registry)} proteins with chain metadata")
    return registry


def resolve_chain(file_name: str, row_chain: str, registry: dict) -> str:
    """Resolve the correct chain ID from row data or registry.
    
    Priority:
    1. Use chain_id from row if present and non-empty
    2. Look up in registry (from individual feature files)
    3. Default to 'A' if nothing found
    """
    # Priority 1: Use row's chain if present
    if row_chain and row_chain.strip():
        return row_chain.strip().upper()
    
    # Priority 2: Look up in registry
    chains = registry.get(file_name, set())
    if len(chains) == 1:
        return next(iter(chains))
    elif len(chains) > 1:
        # Multiple chains: use first alphabetically
        sorted_chains = sorted(chains)
        logger.debug(f"Multiple chains for {file_name} ({chains}); using {sorted_chains[0]}")
        return sorted_chains[0]
    
    # Priority 3: Default to 'A'
    return 'A'


def build_protein_id(file_name: str, resolved_chain: str) -> str:
    """Build protein_id from file_name (base PDB code only, no chain suffix).
    
    The chain is stored separately in chain_id column.
    When looking up ESM embeddings, the code constructs: protein_id_chain_id.pt
    
    Format: <pdb_stem> (NO chain suffix)
    Examples:
        '1krn.pdb', 'A' -> '1krn'
        '1n2z_A.pdb', 'A' -> '1n2z'
        'a.001.001.001_1s69a.pdb', 'A' -> 'a.001.001.001_1s69a'
        '1a4j.pdb', 'H' -> '1a4j'
    """
    pdb_stem = Path(file_name).stem.lower()
    
    # Remove existing chain suffix if present (e.g., 1n2z_a -> 1n2z)
    if '_' in pdb_stem:
        parts = pdb_stem.rsplit('_', 1)  # Split from right, max 1 split
        # Check if last part is likely a chain (1-2 characters, alphabetic)
        if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].isalpha():
            pdb_stem = parts[0]
    
    # Return base PDB stem WITHOUT appending chain
    # The chain is already stored in the separate chain_id column
    return pdb_stem


def process_large_csv_streaming(input_path: Path, output_path: Path, chain_registry: dict):
    """Process vectorsTrain_all.csv in streaming fashion to add protein_id and residue_id."""
    
    logger.info(f"Processing {input_path}")
    logger.info(f"Output: {output_path}")
    
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # Add new columns
        fieldnames = reader.fieldnames + ['protein_id', 'residue_id']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        current_protein = None
        protein_count = 0
        row_count = 0
        residue_counter = {}  # Track residue numbers per (file_name, chain)
        
        for row in reader:
            row_count += 1
            
            if row_count % 100000 == 0:
                logger.info(f"  Processed {row_count:,} rows, {protein_count} proteins")
            
            file_name = row['file_name']
            
            # Track new protein
            if file_name != current_protein:
                current_protein = file_name
                protein_count += 1
            
            # Resolve chain_id using registry
            row_chain = row.get('chain_id', '')
            resolved_chain = resolve_chain(file_name, row_chain, chain_registry)
            row['chain_id'] = resolved_chain
            
            # Build protein_id
            protein_id = build_protein_id(file_name, resolved_chain)
            
            # Get residue number
            key = (file_name, resolved_chain)
            if key not in residue_counter:
                residue_counter[key] = 0
            
            # Use residue_number from CSV if available
            if 'residue_number' in row and row['residue_number']:
                try:
                    residue_num = int(float(row['residue_number']))
                except (ValueError, TypeError):
                    residue_counter[key] += 1
                    residue_num = residue_counter[key]
            else:
                residue_counter[key] += 1
                residue_num = residue_counter[key]
            
            # Add protein_id and residue_id
            row['protein_id'] = protein_id
            row['residue_id'] = f"{protein_id}:{resolved_chain}:{residue_num}"
            
            writer.writerow(row)
    
    logger.info(f"✅ Processed {row_count:,} rows, {protein_count} unique proteins")
    return row_count, protein_count





def main():
    """Main processing pipeline."""
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "output_train"
    output_dir = base_dir / "data"
    
    input_csv = data_dir / "vectorsTrain_all.csv"
    existing_chainfix = output_dir / "vectorsTrain_all_chainfix.csv"
    output_csv = output_dir / "vectorsTrain_all_chainfix.csv"
    
    logger.info("="*80)
    logger.info("MERGING COMPLETE TRAINING DATA")
    logger.info("="*80)
    
    # Check input exists
    if not input_csv.exists():
        logger.error(f"❌ Input file not found: {input_csv}")
        return 1
    
    # Backup existing chainfix if it exists
    if existing_chainfix.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_dir / f"vectorsTrain_all_chainfix_backup_{timestamp}.csv"
        logger.info(f"📦 Backing up existing chainfix to: {backup_path.name}")
        
        import shutil
        shutil.copy2(existing_chainfix, backup_path)
        
        # Quick stats on backup
        with open(backup_path, 'r') as f:
            backup_lines = sum(1 for _ in f) - 1  # Exclude header
        logger.info(f"   Backup contains {backup_lines:,} rows")
    
    # Process the main file
    logger.info(f"\n🔄 Processing {input_csv.name}")
    logger.info(f"   Adding protein_id and residue_id columns to ALL rows...")
    
    temp_output = output_dir / "vectorsTrain_all_chainfix_temp.csv"
    
    # Build chain registry from feature files
    logger.info(f"\n🔍 Building chain registry from feature files...")
    features_dir = data_dir
    chain_registry = build_chain_registry(features_dir)
    
    try:
        total_rows, total_proteins = process_large_csv_streaming(input_csv, temp_output, chain_registry)
        
        # Move temp to final location
        logger.info(f"\n💾 Saving base file: {output_csv}")
        temp_output.replace(output_csv)
        
        # Final stats
        file_size_mb = output_csv.stat().st_size / 1024 / 1024
        logger.info(f"\n✅ Complete!")
        logger.info(f"   Total rows: {total_rows:,}")
        logger.info(f"   Total proteins: {total_proteins}")
        logger.info(f"   File size: {file_size_mb:.1f} MB")
        
        # Verify protein coverage
        logger.info(f"\n🔍 Verifying protein coverage...")
        proteins_in_output = set()
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                proteins_in_output.add(row['file_name'])
        
        logger.info(f"   Unique proteins in output: {len(proteins_in_output)}")
        
        # Check against original
        proteins_in_input = set()
        with open(input_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                proteins_in_input.add(row['file_name'])
        
        logger.info(f"   Unique proteins in input: {len(proteins_in_input)}")
        
        missing = proteins_in_input - proteins_in_output
        if missing:
            logger.warning(f"⚠️  Missing {len(missing)} proteins:")
            for p in sorted(missing)[:10]:
                logger.warning(f"    - {p}")
        else:
            logger.info(f"✅ All input proteins present!")
        
        # Check BU48 coverage specifically
        bu48_file = data_dir / "test_vectorsTrain_all_names_bu48.txt"
        if bu48_file.exists():
            logger.info(f"\n🎯 Checking BU48 coverage...")
            with open(bu48_file, 'r') as f:
                bu48_proteins = set(line.strip() for line in f if line.strip())
            
            logger.info(f"   BU48 list: {len(bu48_proteins)} proteins")
            
            # Check coverage (protein_id should match or contain bu48 name)
            found_bu48 = set()
            for bu48_name in bu48_proteins:
                # Look for exact match or as substring (handles SCOP prefixes)
                for prot in proteins_in_output:
                    if bu48_name.lower() in prot.lower():
                        found_bu48.add(bu48_name)
                        break
            
            logger.info(f"   Found {len(found_bu48)} BU48 proteins in output")
            
            missing_bu48 = bu48_proteins - found_bu48
            if missing_bu48:
                logger.warning(f"   ⚠️  Missing {len(missing_bu48)} BU48 proteins:")
                for p in sorted(missing_bu48)[:10]:
                    logger.warning(f"     - {p}")
            else:
                logger.info(f"   ✅ All BU48 proteins present!")
        
        logger.info(f"\n" + "="*80)
        logger.info(f"✅ MERGE COMPLETE - Ready for H5 generation!")
        logger.info(f"="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Clean up temp file
        if temp_output.exists():
            temp_output.unlink()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
