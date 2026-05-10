# Android TFLite Activity Recognition

This project explores on-device activity recognition using smartphone sensor data and TensorFlow Lite.

## Goal

Build an end-to-end embedded AI pipeline:

1. Train a lightweight activity recognition model.
2. Convert it to TensorFlow Lite.
3. Deploy it in an Android app.
4. Run real-time inference from phone sensors.
5. Measure latency and practical constraints.

## Planned Features

- Sensor-based activity recognition
- TensorFlow Lite inference
- Android real-time sensor pipeline
- Latency benchmarking
- Future: personalization and quantization

## Dataset

Initial version uses the UCI HAR dataset.

Activities:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

Training set:
- 7,352 samples
- 561 engineered features per sample