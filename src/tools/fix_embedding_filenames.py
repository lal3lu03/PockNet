#!/usr/bin/env python3
"""
Fix ESM embedding filenames to match the corrected CSV format.

OLD format: 3iyt_a_A.pt (double chain - from old CSV)
NEW format: 3iyt_A.pt (single chain - matches corrected CSV)

This script:
1. Scans all .pt files in the ESM embedding directory
2. Identifies files with the old double-chain format
3. Renames them to the new single-chain format
4. Creates a backup mapping file for safety
"""

import sys
from pathlib import Path
import shutil
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_filename(filename: str):
    """Parse embedding filename to extract components.
    
    Examples:
        1krn_A.pt → ('1krn', 'A') [CORRECT]
        3iyt_a_A.pt → ('3iyt', 'a', 'A') [OLD FORMAT - needs fix]
        a.001.001.001_1s69a_A.pt → complex SCOP format
    """
    stem = Path(filename).stem  # Remove .pt
    parts = stem.split('_')
    
    # Check if it's the old double-chain format
    if len(parts) >= 3:
        # Last part should be uppercase chain
        last = parts[-1]
        second_last = parts[-2]
        
        # If second-to-last is lowercase 1-2 letter chain, it's old format
        if (len(second_last) <= 2 and 
            second_last.isalpha() and 
            second_last.islower() and
            len(last) <= 2 and
            last.isalpha() and
            last.isupper()):
            
            # Extract: everything except second_last + final chain
            protein_base = '_'.join(parts[:-2])
            old_chain_lower = second_last
            chain_upper = last
            return ('old_format', protein_base, old_chain_lower, chain_upper)
    
    # Not old format
    return ('ok', stem)


def fix_embedding_filenames(emb_dir: Path, dry_run: bool = False):
    """Rename embedding files from old to new format.
    
    Args:
        emb_dir: Directory containing .pt embedding files
        dry_run: If True, only report changes without making them
    """
    
    logger.info("="*80)
    logger.info("FIXING ESM EMBEDDING FILENAMES")
    logger.info("="*80)
    logger.info(f"Directory: {emb_dir}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")
    
    if not emb_dir.exists():
        logger.error(f"❌ Directory not found: {emb_dir}")
        return 1
    
    # Scan all .pt files
    all_files = list(emb_dir.glob("*.pt"))
    logger.info(f"Found {len(all_files)} embedding files")
    
    # Analyze files
    to_rename = []
    already_correct = []
    scop_format = []
    
    for pt_file in all_files:
        result = parse_filename(pt_file.name)
        
        if result[0] == 'old_format':
            _, protein_base, old_chain, chain_upper = result
            new_name = f"{protein_base}_{chain_upper}.pt"
            new_path = emb_dir / new_name
            
            # Check if target already exists
            if new_path.exists():
                logger.warning(f"⚠️  Target exists: {new_name}")
                logger.warning(f"   Source: {pt_file.name}")
                logger.warning(f"   Will skip this file")
            else:
                to_rename.append((pt_file, new_path, protein_base, old_chain, chain_upper))
        
        elif 'a.0' in pt_file.name or 'b.0' in pt_file.name or 'c.0' in pt_file.name:
            scop_format.append(pt_file)
        else:
            already_correct.append(pt_file)
    
    # Report findings
    logger.info("")
    logger.info("📊 ANALYSIS:")
    logger.info(f"  Files needing rename: {len(to_rename)}")
    logger.info(f"  Already correct format: {len(already_correct)}")
    logger.info(f"  SCOP format (may need check): {len(scop_format)}")
    
    if len(to_rename) == 0:
        logger.info("")
        logger.info("✅ No files need renaming - all files already in correct format!")
        return 0
    
    # Show sample renamings
    logger.info("")
    logger.info("📝 SAMPLE RENAMES (first 10):")
    for old_path, new_path, protein_base, old_chain, chain_upper in to_rename[:10]:
        logger.info(f"  {old_path.name:<30} → {new_path.name}")
    
    if len(to_rename) > 10:
        logger.info(f"  ... and {len(to_rename) - 10} more")
    
    # Create backup mapping file and backup directory
    if not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mapping_file = emb_dir / f"rename_mapping_{timestamp}.txt"
        backup_dir = emb_dir / f"backup_{timestamp}"
        
        logger.info("")
        logger.info(f"💾 Creating backup mapping: {mapping_file.name}")
        
        with open(mapping_file, 'w') as f:
            f.write("# ESM Embedding Filename Rename Mapping\n")
            f.write(f"# Created: {datetime.now()}\n")
            f.write(f"# Format: OLD_NAME → NEW_NAME\n")
            f.write("#\n")
            
            for old_path, new_path, protein_base, old_chain, chain_upper in to_rename:
                f.write(f"{old_path.name} → {new_path.name}\n")
        
        logger.info(f"   Saved {len(to_rename)} mappings")
        
        # Create backup copies before renaming
        logger.info("")
        logger.info(f"💾 Creating backup copies in: {backup_dir.name}/")
        backup_dir.mkdir(exist_ok=True)
        
        for old_path, new_path, protein_base, old_chain, chain_upper in to_rename:
            backup_file = backup_dir / old_path.name
            shutil.copy2(old_path, backup_file)
        
        logger.info(f"   Backed up {len(to_rename)} files")
    
    # Perform renames
    success_count = 0
    error_count = 0
    
    if dry_run:
        logger.info("")
        logger.info("🔍 DRY RUN - No files will be modified")
        logger.info("   Remove --dry-run flag to apply changes")
    else:
        logger.info("")
        logger.info("🔄 RENAMING FILES...")
        
        for old_path, new_path, protein_base, old_chain, chain_upper in to_rename:
            try:
                old_path.rename(new_path)
                success_count += 1
                
                if success_count % 50 == 0:
                    logger.info(f"   Renamed {success_count}/{len(to_rename)} files...")
                
            except Exception as e:
                logger.error(f"❌ Failed to rename {old_path.name}: {e}")
                error_count += 1
        
        logger.info("")
        logger.info("📊 RESULTS:")
        logger.info(f"  Successfully renamed: {success_count}")
        logger.info(f"  Errors: {error_count}")
        
        if error_count == 0:
            logger.info("")
            logger.info("✅ ALL FILES RENAMED SUCCESSFULLY!")
        else:
            logger.warning("")
            logger.warning(f"⚠️  {error_count} files had errors - check logs above")
    
    logger.info("")
    logger.info("="*80)
    
    return 0 if error_count == 0 else 1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix ESM embedding filenames to match corrected CSV format"
    )
    parser.add_argument(
        '--emb-dir',
        type=Path,
        default=Path('data/esm2_3B_chain'),
        help='Directory containing ESM embeddings (default: data/esm2_3B_chain)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be renamed without making changes'
    )
    
    args = parser.parse_args()
    
    return fix_embedding_filenames(args.emb_dir, args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
