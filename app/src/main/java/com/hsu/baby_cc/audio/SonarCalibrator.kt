package com.hsu.baby_cc.audio

import android.content.Context
import android.content.SharedPreferences
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sin

data class HardwareHealthResult(
    val isHealthy: Boolean,
    val statusMessage: String
)

class SonarCalibrator(private val context: Context) {

    private val sampleRate = 16000
    private val prefs: SharedPreferences = context.getSharedPreferences("sonar_prefs", Context.MODE_PRIVATE)

    /**
     * Executes Active Hardware Health Check Gate:
     * Plays 0.5s Logarithmic Chirp while simultaneously recording from microphone
     * to verify speaker playback and microphone input loop integrity.
     */
    suspend fun checkHardwareHealth(): HardwareHealthResult = withContext(Dispatchers.IO) {
        val recordedSamples = mutableListOf<Float>()
        val recorder = SafeAudioRecorder(context)

        // 1. Start background recording to listen to chirp
        val recordJob = launch(Dispatchers.IO) {
            try {
                recorder.startRecording().collect { pcmShorts ->
                    pcmShorts.forEach { s ->
                        recordedSamples.add(s / 32768.0f)
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        // Wait 200ms for AudioRecord to warm up
        delay(200)

        // 2. Play Logarithmic Chirp (50Hz - 8000Hz)
        playLogarithmicChirp()

        // Wait 800ms for chirp playback & audio capture
        delay(800)

        // 3. Stop recording
        recordJob.cancel()

        // 4. Analyze recorded audio peak energy to verify mic & speaker hardware loop
        val peak = if (recordedSamples.isNotEmpty()) {
            recordedSamples.maxOf { abs(it) }
        } else 0.0f

        clearGains()

        if (peak >= 0.008f) {
            HardwareHealthResult(
                isHealthy = true,
                statusMessage = "聲納硬體健檢通過：喇叭發聲與麥克風收音運作正常！(響應振幅: ${"%.3f".format(peak)})"
            )
        } else {
            HardwareHealthResult(
                isHealthy = false,
                statusMessage = "聲納硬體健檢異常：未偵測到收音回應，請檢查麥克風權限或保護殼遮擋！"
            )
        }
    }

    /**
     * Generates and plays 0.5s Logarithmic Chirp (50Hz -> 8000Hz)
     */
    private fun playLogarithmicChirp() {
        val duration = 0.5 // seconds
        val numSamples = (duration * sampleRate).toInt()
        val generatedSnd = ShortArray(numSamples)

        val f0 = 50.0 // Hz
        val f1 = 8000.0 // Hz
        val k = (f1 / f0)
        val logK = ln(k)

        for (i in 0 until numSamples) {
            val t = i.toDouble() / sampleRate
            val phase = 2.0 * Math.PI * f0 * (duration / logK) * (Math.pow(k, t / duration) - 1.0)
            val sample = sin(phase)

            // Apply fade-in / fade-out to prevent clicks
            val envelope = when {
                i < 160 -> i / 160.0
                i > numSamples - 160 -> (numSamples - i) / 160.0
                else -> 1.0
            }

            generatedSnd[i] = (sample * envelope * 32767.0).toInt().toShort()
        }

        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(generatedSnd.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        audioTrack.write(generatedSnd, 0, generatedSnd.size)
        audioTrack.play()
    }

    private fun loadReferenceSpectrum(): FloatArray {
        return try {
            val jsonString = context.assets.open("reference_spectrum.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            val jsonArray = jsonObject.getJSONArray("reference_mel_energy")
            FloatArray(jsonArray.length()) { i ->
                jsonArray.getDouble(i).toFloat()
            }
        } catch (e: Exception) {
            FloatArray(80) { 1.0f }
        }
    }

    fun clearGains() {
        prefs.edit().remove("mel_gains").apply()
    }
}
