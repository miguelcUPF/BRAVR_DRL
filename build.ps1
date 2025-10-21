# build.ps1 - Sets up Python + PyTorch and builds ALVR

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
} catch {
    Write-Error "Python is not installed or not on PATH. Please install Python 3.x and try again."
    exit 1
}

Write-Host "Python detected: $pythonVersion"

# Check if torch is installed
$torchInstalled = python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch not found. Installing PyTorch 2.0.0 CPU version..."
    python -m pip install --upgrade pip
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyTorch. Exiting."
        exit 1
    }
    Write-Host "PyTorch installed successfully."
} else {
    Write-Host "PyTorch already installed."
}

# Set environment variable to use Python PyTorch for tch-rs
$env:LIBTORCH_USE_PYTORCH = "1"
Write-Host "Environment variable LIBTORCH_USE_PYTORCH=1 set."

# Build ALVR
Write-Host "Starting ALVR build..."
cargo build

if ($LASTEXITCODE -eq 0) {
    Write-Host "ALVR built successfully!"
} else {
    Write-Error "ALVR build failed."
    exit 1
}

Write-Host "ALVR build complete."