# Android TFLite Activity Recognition

This project explores on-device human activity recognition using smartphone sensor data and TensorFlow Lite.

## Goal

Build a complete embedded AI pipeline:

1. Train a lightweight activity recognition model
2. Convert the model to TensorFlow Lite
3. Optimize the model for edge deployment
4. Deploy the model inside an Android application
5. Run on-device inference and benchmark latency
6. Extend toward real-time sensor streaming and personalization

---

## Dataset

Initial experiments use the UCI HAR (Human Activity Recognition) dataset.

### Activities
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### Dataset Statistics

| Metric | Value |
|---|---:|
| Training samples | 7,352 |
| Test samples | 2,947 |
| Features per sample | 561 |

The dataset contains engineered accelerometer and gyroscope features extracted from smartphone motion signals.

---

## Baseline Model

Model:
- Dense neural network
- Input: 561 engineered sensor features
- Output: 6 activity classes

### Baseline Results

| Metric | Value |
|---|---:|
| Test accuracy | 93.4% |
| FP32 TFLite model size | 316.9 KB |
| Mean latency | 0.0072 ms |
| P95 latency | 0.0080 ms |
| P99 latency | 0.0088 ms |

Note: CPU benchmarks were measured on laptop hardware using TensorFlow Lite runtime.

---

## Quantization

Dynamic range quantization was applied to reduce model size and improve inference speed for edge deployment.

### Quantization Results

| Model | Accuracy | Size | Mean Latency | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| FP32 TFLite | 93.4% | 316.9 KB | 0.0072 ms | 0.0080 ms | 0.0088 ms |
| Dynamic Quantized TFLite | 93.4% | 84.9 KB | 0.0026 ms | 0.0028 ms | 0.0037 ms |

Quantization reduced model size by approximately 73% while preserving accuracy.

---

## Android Deployment

The quantized TensorFlow Lite model was deployed into a native Android application using the TensorFlow Lite Android runtime.

### Android Inference

| Device | Inference Latency |
|---|---:|
| Pixel 5 Android Emulator | ~1.27 ms |

The Android application successfully:
- Loaded the quantized `.tflite` model
- Executed on-device inference
- Returned predicted activities and confidence scores
- Measured inference latency directly on Android

---

## Current Pipeline

```text
Sensor Data
    ↓
Feature Extraction
    ↓
Neural Network Training
    ↓
TensorFlow Lite Conversion
    ↓
Quantization
    ↓
Android On-Device Inference
```

---

## Future Work

- Real-time accelerometer and gyroscope streaming
- Sliding-window activity recognition
- Personalized activity adaptation
- Battery and memory profiling
- Live Android sensor inference
- Edge AI optimization experiments
- Comparison between FP32 and INT8 quantization