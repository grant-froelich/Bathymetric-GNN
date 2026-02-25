@echo off
REM Installation script for bathymetric-gnn (Windows)
REM 
REM Run this from Anaconda Prompt or a terminal with conda available.

echo ==========================================
echo Bathymetric GNN Installation (Windows)
echo ==========================================

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda not found. Please install Anaconda or Miniconda first.
    echo Run this script from Anaconda Prompt.
    exit /b 1
)

REM Create environment
echo.
echo Step 1: Creating conda environment...
conda create -n bathymetric-gnn python=3.11 -y

REM Activate environment
echo.
echo Step 2: Activating environment...
call conda activate bathymetric-gnn

REM Install PyTorch with CUDA
echo.
echo Step 3: Installing PyTorch with CUDA 12.1...
echo (If you have CUDA 11.x, edit this script to use pytorch-cuda=11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

REM Install PyTorch Geometric
echo.
echo Step 4: Installing PyTorch Geometric...
pip install torch-geometric

REM Install geospatial packages
echo.
echo Step 5: Installing geospatial packages...
conda install -c conda-forge gdal rasterio shapely pyproj h5py -y

REM Install remaining dependencies
echo.
echo Step 6: Installing remaining packages...
conda install -c conda-forge numpy scipy scikit-learn pandas matplotlib seaborn tqdm pyyaml -y
pip install wandb tensorboard

REM Verify installation
echo.
echo Step 7: Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')"
python -c "from osgeo import gdal; print(f'GDAL: {gdal.__version__}')"

echo.
echo ==========================================
echo Installation complete!
echo.
echo To activate the environment:
echo   conda activate bathymetric-gnn
echo.
echo To test the pipeline:
echo   python scripts/test_pipeline.py --survey /path/to/survey.bag
echo ==========================================
pause
