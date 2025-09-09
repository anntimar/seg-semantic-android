package com.example.segapp

import android.content.Context
import android.graphics.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

class TFLiteUNet(
    private val context: Context,
    private val size: Int = 256,
    private val threshold: Float = 0.5f
) {
    // mesma normalização usada no treino (ImageNet)
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    private val interpreter: Interpreter by lazy {
        val model = FileUtil.loadMappedFile(context, "model_float32.tflite")
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        Interpreter(model, opts)
    }

    fun segment(inputBmp: Bitmap): Bitmap {
        // garante bitmap "software" em ARGB_8888 (getPixels não funciona em HARDWARE)
        val soft = if (inputBmp.config == Bitmap.Config.HARDWARE || inputBmp.config != Bitmap.Config.ARGB_8888) {
            inputBmp.copy(Bitmap.Config.ARGB_8888, /*isMutable=*/false)
        } else {
            inputBmp
        }

        // redimensiona para o tamanho do modelo (NHWC)
        val resized = Bitmap.createScaledBitmap(soft, size, size, true)

        // entrada NHWC float32 [1, H, W, 3] normalizada
        val inBuf = ByteBuffer.allocateDirect(1 * size * size * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        var p = 0
        for (y in 0 until size) {
            for (x in 0 until size) {
                val c = pixels[p++]
                val r0 = ((c ushr 16) and 0xFF) / 255f
                val g0 = ((c ushr  8) and 0xFF) / 255f
                val b0 = ( c         and 0xFF) / 255f
                val r = (r0 - MEAN[0]) / STD[0]
                val g = (g0 - MEAN[1]) / STD[1]
                val b = (b0 - MEAN[2]) / STD[2]
                inBuf.putFloat(r); inBuf.putFloat(g); inBuf.putFloat(b)
            }
        }
        inBuf.rewind()

        // saída FP32: [1, H, W, 1] (logits) -> aplica sigmoid
        val outBuf = ByteBuffer.allocateDirect(1 * size * size * 1 * 4).order(ByteOrder.nativeOrder())
        interpreter.run(inBuf, outBuf)
        outBuf.rewind()

        val mask = FloatArray(size * size)
        for (i in mask.indices) {
            val logit = outBuf.float
            mask[i] = (1f / (1f + exp(-logit.toDouble()).toFloat()))
        }

        return overlayMask(resized, mask, threshold)
    }

    private fun overlayMask(base: Bitmap, mask: FloatArray, thr: Float): Bitmap {
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val paint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.RED
            alpha = 120  // intensidade do overlay
            isAntiAlias = false
        }
        var i = 0
        for (y in 0 until size) {
            for (x in 0 until size) {
                if (mask[i++] >= thr) {
                    canvas.drawPoint(x.toFloat(), y.toFloat(), paint)
                }
            }
        }
        return out
    }
}
