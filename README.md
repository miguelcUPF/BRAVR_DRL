Our ALVR fork uses `tch-rs` (Rust bindings for PyTorch) for some components. On Windows, you must have **Python** and **PyTorch** (v2.9.0) installed.

To build the project:

1. Open PowerShell in the ALVR project directory.  
2. Run the build script:

    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\build.ps1
    ```

The script will:
- Check that Python is installed.  
- Install PyTorch v2.9.0 if it's missing.  
- Set the `LIBTORCH_USE_PYTORCH` environment variable.  
- Build ALVR using Cargo.