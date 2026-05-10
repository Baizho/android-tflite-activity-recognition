package com.baizho.hartflite

import android.annotation.SuppressLint
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter

    private val activityLabels = listOf(
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    )

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val outputText = TextView(this)
        outputText.textSize = 18f
        outputText.setPadding(32, 64, 32, 32)
        setContentView(outputText)

        interpreter = Interpreter(loadModelFile("har_baseline_dynamic_quant.tflite"))

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
            
            Input shape: [1, 561]
            Output shape: [1, 6]
            
            Dummy prediction: ${activityLabels[predictedIndex]}
            Confidence: ${(confidence * 100).roundToInt()} percent
            Latency: ${"%.4f".format(latencyMs)} ms
        """.trimIndent()
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
        interpreter.close()
        super.onDestroy()
    }
}