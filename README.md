# CameraCalib

Script for chessboard camera calibration. 

## Usage:
```bash
python3 calib_intrinsics.py \
  --srcpath=/path/to/source/photos \
  --outdir=/path/to/save/undistorted/photos \
  --pattern=COLUMNSxROWS \
  --iterations=1 \
  --saveto=calib.json
```

The camera paameters will be written to the file `calib.json`. 
