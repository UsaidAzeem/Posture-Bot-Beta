# Posture Bot Beta

A posture estimation model for medical competitions using MediaPipe and OpenCV. The system analyzes body posture in real-time and provides feedback on alignment quality.

## Features

- Real-time pose detection using MediaPipe
- Multiple analysis metrics:
  - Neck inclination angle
  - Torso inclination angle
  - Head forward position
- Smoothed analysis using moving averages
- Visual skeleton overlay with color-coded feedback
- Configurable thresholds via GUI (dearpygui)

## Requirements

```
pip install -r requirements.txt
```

## Usage

### CLI Version
```bash
python human_posture_analysis_video.py
```

### GUI Version
```bash
python posture_app.py
```

## Files

- `human_posture_analysis_video.py` - CLI version with video output
- `posture_app.py` - GUI version with real-time sliders
- `pose_landmarker_lite.task` - MediaPipe pose landmarker model

## Example Output

![Example](example_output.gif)

## License

MIT