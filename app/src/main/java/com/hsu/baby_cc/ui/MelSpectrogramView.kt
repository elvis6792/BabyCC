package com.hsu.baby_cc.ui

import android.graphics.Bitmap
import android.graphics.Color
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp

@Composable
fun MelSpectrogramView(
    melData: FloatArray?, // Size: 80 * 157
    modifier: Modifier = Modifier
) {
    val bitmap = remember(melData) {
        if (melData == null || melData.size != 80 * 157) {
            Bitmap.createBitmap(157, 80, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        } else {
            createFireSpectrogramBitmap(melData, 80, 157)
        }
    }

    Image(
        bitmap = bitmap.asImageBitmap(),
        contentDescription = "Mel Spectrogram",
        contentScale = ContentScale.FillBounds,
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(157f / 80f)
            .clip(RoundedCornerShape(8.dp))
    )
}

/**
 * Maps dB values [-100.0, 0.0] to [0.0, 1.0] and renders using the "Fire" palette.
 */
private fun createFireSpectrogramBitmap(
    melData: FloatArray,
    nMels: Int,
    targetFrames: Int
): Bitmap {
    val bitmap = Bitmap.createBitmap(targetFrames, nMels, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(nMels * targetFrames)

    // Fire Colormap mapping function
    fun getFireColor(normVal: Float): Int {
        val valClamped = normVal.coerceIn(0.0f, 1.0f)
        val r: Int
        val g: Int
        val b: Int

        when {
            valClamped < 0.33f -> {
                // Black to Red
                val t = valClamped / 0.33f
                r = (t * 255).toInt()
                g = 0
                b = 0
            }
            valClamped < 0.66f -> {
                // Red to Yellow
                val t = (valClamped - 0.33f) / 0.33f
                r = 255
                g = (t * 255).toInt()
                b = 0
            }
            else -> {
                // Yellow to White
                val t = (valClamped - 0.66f) / 0.34f
                r = 255
                g = 255
                b = (t * 255).toInt()
            }
        }
        return Color.rgb(r, g, b)
    }

    // Mel values stored in C++ as row-major [80, 157]
    // In Spectrograms, low frequencies (m = 0) are usually rendered at the bottom!
    for (m in 0 until nMels) {
        val yIdx = nMels - 1 - m // Flip vertically for standard spectrogram orientation
        for (f in 0 until targetFrames) {
            val dbVal = melData[m * targetFrames + f]
            // Scale [-80.0dB, 0.0dB] -> [0.0, 1.0] for Fire palette
            val normVal = (dbVal + 80.0f) / 80.0f
            pixels[yIdx * targetFrames + f] = getFireColor(normVal)
        }
    }

    bitmap.setPixels(pixels, 0, targetFrames, 0, 0, targetFrames, nMels)
    return bitmap
}
