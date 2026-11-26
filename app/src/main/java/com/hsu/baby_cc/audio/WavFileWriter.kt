package com.hsu.baby_cc.audio

import java.io.File
import java.io.RandomAccessFile

class WavFileWriter(file: File, private val sampleRate: Int = 16000) {

    private val raf = RandomAccessFile(file, "rw")
    private var totalPcmBytes = 0

    init {
        // Write placeholder 44-byte WAV header
        raf.write(ByteArray(44))
    }

    fun writePcm(pcmShorts: ShortArray) {
        val byteBuffer = ByteArray(pcmShorts.size * 2)
        for (i in pcmShorts.indices) {
            val s = pcmShorts[i].toInt()
            byteBuffer[i * 2] = (s and 0xFF).toByte()
            byteBuffer[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
        }
        raf.write(byteBuffer)
        totalPcmBytes += byteBuffer.size
    }

    fun close() {
        raf.seek(0)
        val channels = 1
        val byteRate = sampleRate * channels * 2
        val blockAlign = channels * 2

        val header = ByteArray(44)
        // RIFF header
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        val totalDataLen = totalPcmBytes + 36
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
        // fmt chunk
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0 // subchunk1 size
        header[20] = 1; header[21] = 0 // PCM format
        header[22] = channels.toByte(); header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = blockAlign.toByte(); header[33] = 0
        header[34] = 16; header[35] = 0 // bits per sample
        // data chunk
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        header[40] = (totalPcmBytes and 0xff).toByte()
        header[41] = ((totalPcmBytes shr 8) and 0xff).toByte()
        header[42] = ((totalPcmBytes shr 16) and 0xff).toByte()
        header[43] = ((totalPcmBytes shr 24) and 0xff).toByte()

        raf.write(header)
        raf.close()
    }
}
