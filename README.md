# Posture Bot Beta

A real-time posture estimation model for medical competitions using MediaPipe and OpenCV. The system analyzes body posture from video input and provides instant feedback on alignment quality.

## The Problem

Poor posture is a widespread issue, especially for people who spend long hours sitting at desks or working on computers. Bad posture can lead to:
- Chronic back and neck pain
- Reduced lung capacity
- Poor circulation
- Headaches and fatigue

Early detection and correction of poor posture habits is crucial for preventing long-term health issues. This project provides an automated solution to monitor and alert users about postural misalignments in real-time.

## How It Works

1. **Pose Detection**: Uses MediaPipe's Pose Landmarker to detect 33 body landmarks in each frame
2. **Angle Calculation**: Computes three key metrics:
   - **Neck Inclination**: Angle between shoulders and ears
   - **Torso Inclination**: Angle between hips and shoulders  
   - **Head Forward Position**: Horizontal distance between ear and shoulder
3. **Smoothing**: Applies moving average (20-frame window) to reduce jitter
4. **Scoring**: Generates a 0-100 score based on deviation from ideal alignment
5. **Visualization**: Draws skeleton overlay with color-coded feedback (green=good, yellow=warning, red=bad)

## Accuracy

- **Pose Detection**: MediaPipe achieves **99.5%** average precision for pose landmark detection
- **Posture Classification**: 
  - Good Posture: Score > 80
  - Slightly Off: Score 60-80
  - Misaligned: Score < 60

The model uses a weighted scoring system:
- Neck: 35%
- Torso: 45%
- Head forward: 20%

## Features

- Real-time pose detection
- Multiple analysis metrics
- Smoothed analysis using moving averages
- Visual skeleton overlay with color-coded feedback
- CLI and GUI versions available

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
- `posture_app.py` - GUI version with real-time threshold adjustment
- `pose_landmarker_lite.task` - MediaPipe pose landmarker model
- `input.mp4` - Sample input video
- `example_output.mp4` - Processed output example

## Example Output

The `example_output.mp4` demonstrates the system detecting posture in real-time, showing:
- Skeleton overlay on the detected person
- Status indicator (ALIGNED/SLIGHTLY OFF/MISALIGNED)
- Real-time score calculation

## License

MIT