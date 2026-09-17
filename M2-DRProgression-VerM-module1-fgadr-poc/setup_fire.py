"""
Step 1: FIRE Dataset Setup
Handles .7z extraction for FIRE dataset
"""

import os
import sys
import subprocess
from pathlib import Path


def install_py7zr():
    """Install py7zr if needed"""
    print("📦 Installing py7zr (for 7z extraction)...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'py7zr'])
    print("✓ py7zr installed")


def setup_fire_dataset(archive_path='./FIRE.7z', extract_to='./FIRE_dataset'):
    """Extract and organize FIRE dataset (handles .7z format)"""
    
    print("="*70)
    print("FIRE DATASET SETUP")
    print("="*70)
    
    # Check for .7z or .zip
    if not os.path.exists(archive_path):
        zip_alt = archive_path.replace('.7z', '.zip')
        if os.path.exists(zip_alt):
            archive_path = zip_alt
            print(f"Found .zip instead: {archive_path}")
        else:
            print(f"\n❌ FIRE archive not found at: {archive_path}")
            print("\n📥 Download instructions:")
            print("1. Go to: https://projects.ics.forth.gr/cvrl/fire/")
            print("2. Fill in the request form")
            print("3. Save the file as 'FIRE.7z' in your project root")
            print("4. Re-run this script")
            return False
    
    # Extract if not already extracted
    if not os.path.exists(extract_to):
        print(f"\n📦 Extracting {archive_path}...")
        
        if archive_path.endswith('.7z'):
            try:
                import py7zr
            except ImportError:
                install_py7zr()
                import py7zr
            
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(extract_to)
            print(f"✓ Extracted to {extract_to}")
        
        elif archive_path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✓ Extracted to {extract_to}")
    else:
        print(f"\n✓ Already extracted at {extract_to}")
    
    # Inspect structure
    print("\n📊 Inspecting FIRE structure...")
    for root, dirs, files in os.walk(extract_to):
        level = root.replace(extract_to, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:
            sub_indent = ' ' * 2 * (level + 1)
            for file in files[:3]:
                print(f'{sub_indent}{file}')
            if len(files) > 3:
                print(f'{sub_indent}... and {len(files)-3} more')
    
    print("\n✓ FIRE dataset is ready!")
    return True


if __name__ == '__main__':
    setup_fire_dataset()