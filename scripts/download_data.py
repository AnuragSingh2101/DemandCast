#!/usr/bin/env python
"""
Download DataCo SMART SUPPLY CHAIN dataset from Mendeley Data.

Dataset: https://data.mendeley.com/datasets/8gx2fvg2k6/5
License: CC BY 4.0
Citation: Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019)
"""

import sys
from pathlib import Path
import urllib.request
import ssl

# Mendeley Data files - these URLs may require authentication
# If automatic download fails, manually download from the URL above
FILES = {
    "DataCoSupplyChainDataset.csv": "https://data.mendeley.com/datasets/8gx2fvg2k6/5/files/1f7b7e4c-3b3c-4e4a-9b8c-5e8f7a6b5c4d/DataCoSupplyChainDataset.csv?dl=1",
    "DescriptionDataCoSupplyChain.csv": "https://data.mendeley.com/datasets/8gx2fvg2k6/5/files/1f7b7e4c-3b3c-4e4a-9b8c-5e8f7a6b5c4d/DescriptionDataCoSupplyChain.csv?dl=1",
}

def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        # Create SSL context that doesn't verify (for problematic certs)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("DemandCast Data Download Script")
    print("=" * 40)
    print("Dataset: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS")
    print("Source: https://data.mendeley.com/datasets/8gx2fvg2k6/5")
    print("License: CC BY 4.0")
    print("Citation: Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019)")
    print()
    
    dest_dir = Path("data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    for filename, url in FILES.items():
        dest_path = dest_dir / filename
        print(f"Downloading {filename}...")
        
        if download_file(url, dest_path):
            size = dest_path.stat().st_size
            print(f"  ✓ Downloaded: {filename} ({size:,} bytes)")
        else:
            print(f"  ✗ Failed to download {filename}")
            all_success = False
    
    print()
    print("Verification:")
    for filename in FILES.keys():
        path = dest_dir / filename
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {filename} exists ({size:,} bytes)")
        else:
            print(f"  ✗ {filename} MISSING")
            all_success = False
    
    if not all_success:
        print()
        print("MANUAL DOWNLOAD REQUIRED:")
        print("1. Go to: https://data.mendeley.com/datasets/8gx2fvg2k6/5")
        print("2. Click 'Download' for each file")
        print("3. Place files in: data/raw/")
        print("   - DataCoSupplyChainDataset.csv")
        print("   - DescriptionDataCoSupplyChain.csv")
        sys.exit(1)
    
    print()
    print("All files downloaded successfully!")
    print("Next step: Run the data audit notebook")
    print("  jupyter lab notebooks/01_data_audit.ipynb")

if __name__ == "__main__":
    main()