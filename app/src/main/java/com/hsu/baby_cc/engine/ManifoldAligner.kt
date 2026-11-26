package com.hsu.baby_cc.engine

import kotlin.math.cos

class ManifoldAligner {

    companion object {
        private const val KERNEL_SIZE = 65

        fun generateFIRFromMelGains(melGains: FloatArray): FloatArray {
            if (melGains.size != 80) {
                throw IllegalArgumentException("Mel gains must have exactly 80 dimensions")
            }

            // 1. Map 80 Mel gains to 128 linear spectrum bins
            val linearSpectrum = FloatArray(128) { 1.0f }
            for (k in 0 until 64) {
                val melIdx = (k * 80) / 64
                linearSpectrum[k] = melGains[melIdx.coerceIn(0, 79)]
            }

            // 2. IDFT to time-domain with Hamming Window
            val firKernel = FloatArray(KERNEL_SIZE)
            val center = KERNEL_SIZE / 2
            var sumCoeffs = 0.0f

            for (i in 0 until KERNEL_SIZE) {
                var sum = 0.0f
                for (k in 0 until 64) {
                    val phase = 2.0 * Math.PI * k * (i - center) / 128.0
                    sum += linearSpectrum[k] * cos(phase).toFloat()
                }

                // Hamming Window
                val window = 0.54 - 0.46 * cos(2.0 * Math.PI * i / (KERNEL_SIZE - 1))
                firKernel[i] = (sum * window).toFloat()
                sumCoeffs += firKernel[i]
            }

            // 3. Normalize kernel so total DC gain = 1.0 (Prevents 64x audio overdrive & clipping)
            if (abs(sumCoeffs) > 1e-6f) {
                for (i in 0 until KERNEL_SIZE) {
                    firKernel[i] /= sumCoeffs
                }
            }

            return firKernel
        }

        private fun abs(value: Float): Float = if (value < 0) -value else value
    }
}
