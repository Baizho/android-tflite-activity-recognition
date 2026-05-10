package com.baizho.hartflite

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var interpreter: Interpreter
    private lateinit var sensorManager: SensorManager
    private lateinit var outputText: TextView

    private var latestAccel = FloatArray(3) { 0f }
    private var latestGyro = FloatArray(3) { 0f }

    private var accelSamples = 0
    private var gyroSamples = 0
    private var combinedSamples = 0

    private val windowSize = 128
    private val sensorWindow = ArrayDeque<FloatArray>()

    private val activityLabels = listOf(
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        outputText = TextView(this)
        outputText.textSize = 16f
        outputText.setPadding(32, 64, 32, 32)
        setContentView(outputText)

        interpreter = Interpreter(loadModelFile("har_baseline_dynamic_quant.tflite"))
        runDummyInference()

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (accelerometer == null || gyroscope == null) {
            outputText.text = "Required sensors not available on this device/emulator."
            return
        }

        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME)
    }

    private fun runDummyInference() {
        val input = Array(1) { FloatArray(561) { 0.0f } }
        val output = Array(1) { FloatArray(6) }

        val start = System.nanoTime()
        interpreter.run(input, output)
        val end = System.nanoTime()

        val predictedIndex = output[0].indices.maxBy { output[0][it] }
        val confidence = output[0][predictedIndex]
        val latencyMs = (end - start) / 1_000_000.0

        outputText.text = """
            TFLite model loaded successfully.

            Dummy prediction: ${activityLabels[predictedIndex]}
            Confidence: ${(confidence * 100).roundToInt()} percent
            Latency: ${"%.4f".format(latencyMs)} ms

            Waiting for live sensor data...
        """.trimIndent()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAccel = event.values.clone()
                accelSamples++
                addCombinedSample()
            }

            Sensor.TYPE_GYROSCOPE -> {
                latestGyro = event.values.clone()
                gyroSamples++
            }
        }

        updateSensorDisplay()
    }

    private fun addCombinedSample() {
        val sample = floatArrayOf(
            latestAccel[0], latestAccel[1], latestAccel[2],
            latestGyro[0], latestGyro[1], latestGyro[2]
        )

        sensorWindow.addLast(sample)
        combinedSamples++

        if (sensorWindow.size > windowSize) {
            sensorWindow.removeFirst()
        }
    }

    private fun updateSensorDisplay() {
        val windowReady = sensorWindow.size == windowSize

        outputText.text = """
            Live Sensor Stream + Sliding Window

            Accelerometer:
            x = ${"%.3f".format(latestAccel[0])}
            y = ${"%.3f".format(latestAccel[1])}
            z = ${"%.3f".format(latestAccel[2])}
            samples = $accelSamples

            Gyroscope:
            x = ${"%.3f".format(latestGyro[0])}
            y = ${"%.3f".format(latestGyro[1])}
            z = ${"%.3f".format(latestGyro[2])}
            samples = $gyroSamples

            Combined samples = $combinedSamples
            Window size = ${sensorWindow.size} / $windowSize
            Window ready = $windowReady

            Current model expects 561 engineered UCI HAR features.
            Next step: train raw-window model with input [128, 6].
        """.trimIndent()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not used yet
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val modelBytes = assets.open(filename).use { it.readBytes() }

        val buffer = ByteBuffer.allocateDirect(modelBytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)
        buffer.rewind()

        return buffer
    }

    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        interpreter.close()
        super.onDestroy()
    }
}