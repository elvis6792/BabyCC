package com.hsu.baby_cc.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext

class SafeAudioRecorder(private val context: Context) {

    private val sampleRate = 16000
    private val bufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    fun startRecording(): Flow<ShortArray> = flow {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            throw SecurityException("Record Audio permission not granted")
        }

        val preferredSources = listOf(
            MediaRecorder.AudioSource.UNPROCESSED,
            MediaRecorder.AudioSource.VOICE_RECOGNITION
        )

        var audioRecord: AudioRecord? = null
        for (source in preferredSources) {
            try {
                audioRecord = AudioRecord(
                    source,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize
                )
                if (audioRecord.state == AudioRecord.STATE_INITIALIZED) {
                    break
                }
            } catch (e: Exception) {
                audioRecord?.release()
                audioRecord = null
            }
        }

        if (audioRecord == null || audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            throw IllegalStateException("Failed to initialize AudioRecord")
        }

        val sessionId = audioRecord.audioSessionId
        disableAllSystemAudioEffects(sessionId)

        audioRecord.startRecording()

        val buffer = ShortArray(bufferSize / 2) // short is 2 bytes
        try {
            while (coroutineContext.isActive) {
                val readSize = audioRecord.read(buffer, 0, buffer.size)
                if (readSize > 0) {
                    emit(buffer.copyOf(readSize))
                }
            }
        } finally {
            audioRecord.stop()
            audioRecord.release()
        }
    }.flowOn(Dispatchers.IO)

    private fun disableAllSystemAudioEffects(sessionId: Int) {
        if (NoiseSuppressor.isAvailable()) {
            NoiseSuppressor.create(sessionId)?.enabled = false
        }
        if (AutomaticGainControl.isAvailable()) {
            AutomaticGainControl.create(sessionId)?.enabled = false
        }
        if (AcousticEchoCanceler.isAvailable()) {
            AcousticEchoCanceler.create(sessionId)?.enabled = false
        }
    }
}
