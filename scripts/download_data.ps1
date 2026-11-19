<#
.SYNOPSIS
    Download DataCo SMART SUPPLY CHAIN dataset from Mendeley Data

.DESCRIPTION
    This script downloads the DataCo dataset files needed for DemandCast.
    Dataset: https://data.mendeley.com/datasets/8gx2fvg2k6/5
    License: CC BY 4.0
    Citation: Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019)

.NOTES
    Mendeley Data may require authentication or have download restrictions.
    If automatic download fails, manually download from the URL above and place files in data/raw/
#>

param(
    [string]$DestinationPath = "data/raw"
)

$urls = @{
    "DataCoSupplyChainDataset.csv" = "https://data.mendeley.com/datasets/8gx2fvg2k6/5/files/1f7b7e4c-3b3c-4e4a-9b8c-5e8f7a6b5c4d/DataCoSupplyChainDataset.csv?dl=1"
    "DescriptionDataCoSupplyChain.csv" = "https://data.mendeley.com/datasets/8gx2fvg2k6/5/files/1f7b7e4c-3b3c-4e4a-9b8c-5e8f7a6b5c4d/DescriptionDataCoSupplyChain.csv?dl=1"
}

Write-Host "DemandCast Data Download Script" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dataset: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS"
Write-Host "Source: https://data.mendeley.com/datasets/8gx2fvg2k6/5"
Write-Host "License: CC BY 4.0"
Write-Host "Citation: Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019)"
Write-Host ""

# Create destination directory
if (-not (Test-Path $DestinationPath)) {
    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    Write-Host "Created directory: $DestinationPath"
}

foreach ($file in $urls.Keys) {
    $url = $urls[$file]
    $outputPath = Join-Path $DestinationPath $file
    
    Write-Host "Downloading $file..." -ForegroundColor Yellow
    
    try {
        Invoke-WebRequest -Uri $url -OutFile $outputPath -UseBasicParsing
        $size = (Get-Item $outputPath).Length
        Write-Host "  ✓ Downloaded: $file ($size bytes)" -ForegroundColor Green
    }
    catch {
        Write-Host "  ✗ Failed to download $file" -ForegroundColor Red
        Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        Write-Host "MANUAL DOWNLOAD REQUIRED:" -ForegroundColor Cyan
        Write-Host "1. Go to: https://data.mendeley.com/datasets/8gx2fvg2k6/5"
        Write-Host "2. Click 'Download' for each file"
        Write-Host "3. Place files in: $DestinationPath"
        Write-Host "   - DataCoSupplyChainDataset.csv"
        Write-Host "   - DescriptionDataCoSupplyChain.csv"
        Write-Host ""
    }
}

Write-Host ""
Write-Host "Verification:" -ForegroundColor Green
foreach ($file in $urls.Keys) {
    $path = Join-Path $DestinationPath $file
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        Write-Host "  ✓ $file exists ($size bytes)"
    }
    else {
        Write-Host "  ✗ $file MISSING"
    }
}

Write-Host ""
Write-Host "Next step: Run the data audit notebook"
Write-Host "  jupyter lab notebooks/01_data_audit.ipynb"