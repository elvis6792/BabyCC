package com.hsu.baby_cc.audio

import kotlin.math.tanh

class StethoAudioProcessor {

    private var lastInput = 0.0f
    private var lastOutput = 0.0f

    // 80Hz cutoff at 16000Hz sample rate: alpha = 1 / (1 + 2*PI*80/16000) = 0.9695f
    private val alpha = 0.9695f

    fun processStethoChunk(pcmShorts: ShortArray): ShortArray {
        val processed = ShortArray(pcmShorts.size)
        val gain = 2.5f // +8 dB gain boost for quiet lung sounds

        for (i in pcmShorts.indices) {
            val input = pcmShorts[i] / 32768.0f

            // 1. 80Hz High-Pass Filter (Cuts off < 80Hz hand shaking, DC offset & rumble)
            val output = alpha * (lastOutput + input - lastInput)
            lastInput = input
            lastOutput = output

            // 2. Gain Boost (+8 dB)
            var boosted = output * gain

            // 3. Soft Limiter (Smooth tanh clamping to prevent clipping/pops)
            if (boosted > 0.8f) {
                boosted = 0.8f + 0.2f * tanh(((boosted - 0.8f) / 0.2f).toDouble()).toFloat()
            } else if (boosted < -0.8f) {
                boosted = -0.8f - 0.2f * tanh(((-boosted - 0.8f) / 0.2f).toDouble()).toFloat()
            }

            val clamped = boosted.coerceIn(-1.0f, 1.0f)
            processed[i] = (clamped * 32767.0f).toInt().toShort()
        }
        return processed
    }

    fun reset() {
        lastInput = 0.0f
        lastOutput = 0.0f
    }
}
