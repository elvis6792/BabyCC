package com.hsu.baby_cc.ui

import android.app.Application
import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hsu.baby_cc.NativeAnalyzer
import com.hsu.baby_cc.audio.SafeAudioRecorder
import com.hsu.baby_cc.audio.SonarCalibrator
import com.hsu.baby_cc.audio.StethoAudioProcessor
import com.hsu.baby_cc.audio.WavFileWriter
import com.hsu.baby_cc.data.AnalysisRecord
import com.hsu.baby_cc.data.AppDatabase
import com.hsu.baby_cc.engine.AcousticDecisionEngine
import com.hsu.baby_cc.engine.AlertLevel
import com.hsu.baby_cc.engine.MonitoringMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class RecordingState {
    IDLE,
    CALIBRATING,
    RECORDING,
    ANALYZING,
    FINISHED
}

data class UiState(
    val mode: MonitoringMode = MonitoringMode.PHONE,
    val state: RecordingState = RecordingState.IDLE,
    val wheezeProb: Float = 0.0f,
    val coughProb: Float = 0.0f,
    val alertLevel: AlertLevel = AlertLevel.NORMAL,
    val consecutiveHits: Int = 0,
    val targetHits: Int = 3,
    val statusMessage: String = "準備就緒",
    val hasTransientSqueak: Boolean = false,
    val melSpectrogram: FloatArray? = null,
    val historyRecords: List<AnalysisRecord> = emptyList(),
    val playingAudioPath: String? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val recorder = SafeAudioRecorder(application)
    private val calibrator = SonarCalibrator(application)
    private val stethoProcessor = StethoAudioProcessor()
    private val nativeAnalyzer = NativeAnalyzer()
    private val db = AppDatabase.getDatabase(application)

    private var decisionEngine = AcousticDecisionEngine(_uiState.value.mode)
    private var recordingJob: Job? = null
    private var wavWriter: WavFileWriter? = null
    private var currentWavFile: File? = null
    private var mediaPlayer: MediaPlayer? = null

    init {
        val success = nativeAnalyzer.initModel(application.assets, "asthma_monitor_final_merged.onnx")
        if (!success) {
            _uiState.value = _uiState.value.copy(
                statusMessage = "AI 模型初始化失敗"
            )
        }

        // Guarantee 100% pure raw audio for AI (Clear legacy FIR filters)
        calibrator.clearGains()

        loadHistory()
    }

    fun setMode(mode: MonitoringMode) {
        if (_uiState.value.state == RecordingState.RECORDING || _uiState.value.state == RecordingState.ANALYZING) return
        decisionEngine = AcousticDecisionEngine(mode)
        _uiState.value = _uiState.value.copy(mode = mode)
    }

    fun startSonarCalibration() {
        if (_uiState.value.state == RecordingState.RECORDING || _uiState.value.state == RecordingState.ANALYZING) return

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(
                state = RecordingState.CALIBRATING,
                statusMessage = "進行聲納麥克風硬體健檢中..."
            )

            val healthResult = calibrator.checkHardwareHealth()

            _uiState.value = _uiState.value.copy(
                state = RecordingState.IDLE,
                statusMessage = healthResult.statusMessage
            )
        }
    }

    fun toggleRecording() {
        if (_uiState.value.state == RecordingState.RECORDING) {
            stopRecording()
        } else if (_uiState.value.state != RecordingState.ANALYZING && _uiState.value.state != RecordingState.CALIBRATING) {
            startRecording()
        }
    }

    private fun startRecording() {
        stopAudio()
        decisionEngine.reset()

        // Prepare WAV File Writer
        val recDir = File(getApplication<Application>().filesDir, "recordings")
        if (!recDir.exists()) recDir.mkdirs()

        val timeStamp = System.currentTimeMillis()
        val wavFile = File(recDir, "audio_$timeStamp.wav")
        currentWavFile = wavFile
        wavWriter = WavFileWriter(wavFile, 16000)

        _uiState.value = _uiState.value.copy(
            state = RecordingState.RECORDING,
            statusMessage = "正在錄音與即時聲學分析...",
            alertLevel = AlertLevel.NORMAL,
            consecutiveHits = 0,
            wheezeProb = 0.0f,
            coughProb = 0.0f
        )

        recordingJob = viewModelScope.launch(Dispatchers.IO) {
            val channel = Channel<ShortArray>(Channel.UNLIMITED)

            // Asynchronous AI Inference Worker (Decoupled from Audio Recording Thread)
            val aiWorkerJob = launch(Dispatchers.Default) {
                val audioBuffer = mutableListOf<Float>()
                val windowSize = 40000 // 2.5 seconds
                val hopSize = 16000    // 1.0 second hop

                for (pcmChunk in channel) {
                    pcmChunk.forEach { sample ->
                        audioBuffer.add(sample / 32768.0f)
                    }

                    while (audioBuffer.size >= windowSize) {
                        val windowData = audioBuffer.subList(0, windowSize).toFloatArray()

                        // Heavy C++ STFT & ONNX Inference runs asynchronously here without blocking AudioRecord!
                        val result = nativeAnalyzer.processAudioWithMel(windowData)
                        if (result != null && result.size >= 2) {
                            val wheezeProb = result[0]
                            val coughProb = result[1]
                            val melData = if (result.size > 2) result.copyOfRange(2, result.size) else null

                            android.util.Log.d("BabyCC_AI", "Inference Window -> Wheeze(w): ${"%.4f".format(wheezeProb)}, Cough(c): ${"%.4f".format(coughProb)}")

                            decisionEngine.processWindow(wheezeProb, coughProb)
                            val decision = decisionEngine.evaluate()

                            _uiState.value = _uiState.value.copy(
                                wheezeProb = wheezeProb,
                                coughProb = coughProb,
                                alertLevel = decision.level,
                                consecutiveHits = decision.consecutiveHits,
                                targetHits = decision.targetHits,
                                hasTransientSqueak = decision.hasTransientSqueak,
                                melSpectrogram = melData
                            )
                        }

                        // Sliding window shift
                        audioBuffer.subList(0, hopSize.coerceAtMost(audioBuffer.size)).clear()
                    }
                }
            }

            try {
                recorder.startRecording().collect { pcmShorts ->
                    // 1. Save 100% pure raw audio to WAV file
                    wavWriter?.writePcm(pcmShorts)

                    // 2. Send 100% pure raw audio chunk to AI Channel asynchronously
                    channel.trySend(pcmShorts)
                }
            } finally {
                channel.close()
                // 等待 AI Worker 處理完佇列中所有已錄入的音訊緩衝，確保 100% 算完不遺漏
                aiWorkerJob.join()
            }
        }
    }

    private fun stopRecording() {
        if (_uiState.value.state != RecordingState.RECORDING) return

        _uiState.value = _uiState.value.copy(
            state = RecordingState.ANALYZING,
            statusMessage = "AI 正在完成最終會話層分析與彙整..."
        )

        viewModelScope.launch(Dispatchers.IO) {
            val job = recordingJob
            recordingJob = null
            job?.cancelAndJoin() // 停止麥克風收音，並等待背景 AI 佇列 100% 運算完成

            // 展示分析中提示，讓使用者明確感知背景運算
            kotlinx.coroutines.delay(1000)

            try {
                wavWriter?.close()
            } catch (e: Exception) {
                e.printStackTrace()
            }
            wavWriter = null

            val decision = decisionEngine.evaluate()
            val triggeredAlarm = decision.level == AlertLevel.ALARM

            val currentRecord = AnalysisRecord(
                wheezeProb = _uiState.value.wheezeProb,
                coughProb = _uiState.value.coughProb,
                triggeredAlarm = triggeredAlarm,
                audioFilePath = currentWavFile?.absolutePath
            )

            db.analysisDao().insert(currentRecord)
            loadHistory()

            val statusMsg = when {
                decision.level == AlertLevel.ALARM -> "氣喘發作警報！建議就醫。"
                decision.level == AlertLevel.WARNING -> "偵測到疑似氣道狹窄音，請持續留意。"
                decision.hasTransientSqueak -> "捕捉到短促咳嗽尾音 (Squeak <0.2s)，未達到氣喘發作門檻。"
                else -> "檢測正常，呼吸道平穩。"
            }

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    state = RecordingState.FINISHED,
                    alertLevel = decision.level,
                    statusMessage = statusMsg
                )
            }
        }
    }

    fun playAudio(path: String) {
        if (_uiState.value.playingAudioPath == path) {
            stopAudio()
            return
        }

        stopAudio()

        try {
            val file = File(path)
            if (!file.exists()) return

            mediaPlayer = MediaPlayer().apply {
                setDataSource(path)
                prepare()
                start()
                setOnCompletionListener {
                    stopAudio()
                }
            }
            _uiState.value = _uiState.value.copy(playingAudioPath = path)
        } catch (e: Exception) {
            e.printStackTrace()
            stopAudio()
        }
    }

    fun stopAudio() {
        try {
            mediaPlayer?.stop()
            mediaPlayer?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        mediaPlayer = null
        _uiState.value = _uiState.value.copy(playingAudioPath = null)
    }

    private fun loadHistory() {
        viewModelScope.launch(Dispatchers.IO) {
            val records = db.analysisDao().getAllRecords()
            _uiState.value = _uiState.value.copy(historyRecords = records)
        }
    }

    fun deleteRecord(record: AnalysisRecord) {
        viewModelScope.launch(Dispatchers.IO) {
            db.analysisDao().delete(record)
            record.audioFilePath?.let { path ->
                val file = File(path)
                if (file.exists()) file.delete()
            }
            loadHistory()
        }
    }

    fun clearAllHistory() {
        viewModelScope.launch(Dispatchers.IO) {
            val records = db.analysisDao().getAllRecords()
            records.forEach { record ->
                record.audioFilePath?.let { path ->
                    val file = File(path)
                    if (file.exists()) file.delete()
                }
            }
            db.analysisDao().deleteAll()
            loadHistory()
        }
    }

    fun shareRecord(context: Context, record: AnalysisRecord) {
        val path = record.audioFilePath ?: return
        val file = File(path)
        if (!file.exists()) return

        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        context.startActivity(Intent.createChooser(intent, "分享檢測音訊檔"))
    }

    override fun onCleared() {
        super.onCleared()
        stopAudio()
        nativeAnalyzer.release()
    }
}
