package com.hsu.baby_cc.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.hsu.baby_cc.data.AnalysisRecord
import com.hsu.baby_cc.engine.AlertLevel
import com.hsu.baby_cc.engine.MonitoringMode
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.toggleRecording()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    primary = Color(0xFF1E88E5),
                    secondary = Color(0xFF26A69A),
                    background = Color(0xFFF5F7FA)
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    BabyCCApp(
                        viewModel = viewModel,
                        onRequestPermission = {
                            if (ContextCompat.checkSelfPermission(
                                    this,
                                    Manifest.permission.RECORD_AUDIO
                                ) == PackageManager.PERMISSION_GRANTED
                            ) {
                                viewModel.toggleRecording()
                            } else {
                                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            }
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BabyCCApp(
    viewModel: MainViewModel,
    onRequestPermission: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()
    val context = LocalContext.current

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "BabyCC 氣喘監測儀 V2.0",
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF1976D2)
                )
            )
        },
        bottomBar = {
            Surface(
                color = MaterialTheme.colorScheme.background,
                tonalElevation = 8.dp,
                shadowElevation = 8.dp
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    ActionButtons(
                        state = uiState.state,
                        onToggleRecording = onRequestPermission,
                        onSonarCalibrate = { viewModel.startSonarCalibration() }
                    )
                }
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(scrollState)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // 1. Status Banner (3-level Alert Card + Analyzing Indicator)
            AlarmStatusCard(
                alertLevel = uiState.alertLevel,
                consecutiveHits = uiState.consecutiveHits,
                targetHits = uiState.targetHits,
                mode = uiState.mode,
                statusMsg = uiState.statusMessage,
                state = uiState.state
            )

            // 2. Mode Selection Tabs
            ModeSelector(
                selectedMode = uiState.mode,
                onModeSelected = { viewModel.setMode(it) },
                enabled = uiState.state != RecordingState.RECORDING
            )

            // 3. Real-time Probabilities Gauge
            ProbabilityMeters(
                wheezeProb = uiState.wheezeProb,
                coughProb = uiState.coughProb,
                mode = uiState.mode
            )

            // 4. Mel-Spectrogram Live Canvas
            Text(
                "即時 Mel-Spectrogram (Fire 視圖)",
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color = Color.Gray
            )

            Box(modifier = Modifier.fillMaxWidth()) {
                MelSpectrogramView(
                    melData = uiState.melSpectrogram,
                    modifier = Modifier.fillMaxWidth()
                )

                androidx.compose.animation.AnimatedVisibility(
                    visible = uiState.hasTransientSqueak,
                    enter = fadeIn() + slideInHorizontally(initialOffsetX = { it / 2 }),
                    exit = fadeOut() + slideOutHorizontally(targetOffsetX = { it / 2 }),
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                ) {
                    Surface(
                        color = Color.Black.copy(alpha = 0.75f),
                        shape = RoundedCornerShape(20.dp),
                        border = BorderStroke(1.dp, Color(0xFFFFC107)),
                        shadowElevation = 4.dp
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                "捕捉到極短促過渡音 (Squeak)",
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFFFFD54F),
                                maxLines = 1
                            )
                        }
                    }
                }
            }

            // 5. Recent Analysis History
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "最近檢測紀錄",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )

                if (uiState.historyRecords.isNotEmpty()) {
                    var showClearDialog by remember { mutableStateOf(false) }

                    TextButton(onClick = { showClearDialog = true }) {
                        Text(
                            "清除全部",
                            color = Color(0xFFD32F2F),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }

                    if (showClearDialog) {
                        AlertDialog(
                            onDismissRequest = { showClearDialog = false },
                            title = { Text("確認清除歷史紀錄？", fontWeight = FontWeight.Bold) },
                            text = { Text("將會永久刪除所有歷史檢測數據以及對應的錄音檔案，此操作無法復原。") },
                            confirmButton = {
                                TextButton(
                                    onClick = {
                                        showClearDialog = false
                                        viewModel.clearAllHistory()
                                    }
                                ) {
                                    Text("確認刪除", color = Color(0xFFD32F2F), fontWeight = FontWeight.Bold)
                                }
                            },
                            dismissButton = {
                                TextButton(onClick = { showClearDialog = false }) {
                                    Text("取消")
                                }
                            }
                        )
                    }
                }
            }

            HistoryList(
                records = uiState.historyRecords,
                playingPath = uiState.playingAudioPath,
                onPlay = { viewModel.playAudio(it) },
                onShare = { viewModel.shareRecord(context, it) },
                onDelete = { viewModel.deleteRecord(it) }
            )

            // Bottom spacing to ensure history is not obscured by bottom bar
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
fun AlarmStatusCard(
    alertLevel: AlertLevel,
    consecutiveHits: Int,
    targetHits: Int,
    mode: MonitoringMode,
    statusMsg: String,
    state: RecordingState
) {
    val bgColor = when {
        state == RecordingState.ANALYZING -> Color(0xFFFFF3E0)
        alertLevel == AlertLevel.ALARM -> Color(0xFFFFEBEE)
        alertLevel == AlertLevel.WARNING -> Color(0xFFFFF8E1)
        else -> Color(0xFFE8F5E9)
    }
    val textColor = when {
        state == RecordingState.ANALYZING -> Color(0xFFE65100)
        alertLevel == AlertLevel.ALARM -> Color(0xFFD32F2F)
        alertLevel == AlertLevel.WARNING -> Color(0xFFF57F17)
        else -> Color(0xFF2E7D32)
    }
    val titleText = when {
        state == RecordingState.ANALYZING -> "AI 會話分析與彙整中..."
        alertLevel == AlertLevel.ALARM -> "氣喘發作警報 (ALARM)"
        alertLevel == AlertLevel.WARNING -> "偵測到疑似氣道狹窄音"
        else -> "氣道狀態正常"
    }

    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = bgColor),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = titleText,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = textColor
            )

            if (state == RecordingState.ANALYZING) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth(0.8f)
                        .height(4.dp),
                    color = Color(0xFFFB8C00),
                    trackColor = Color(0xFFFFE0B2)
                )
            } else {
                if (mode == MonitoringMode.STETHO) {
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "Stetho 模式進度: 已捕捉 $consecutiveHits / $targetHits 個喘鳴視窗",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        color = textColor
                    )
                } else {
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "Phone 模式進度: 已捕捉 $consecutiveHits 個喘鳴視窗",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        color = textColor
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = statusMsg,
                fontSize = 13.sp,
                fontWeight = if (state == RecordingState.ANALYZING) FontWeight.SemiBold else FontWeight.Normal,
                color = if (state == RecordingState.ANALYZING) Color(0xFFE65100) else Color.DarkGray
            )
        }
    }
}

@Composable
fun ModeSelector(
    selectedMode: MonitoringMode,
    onModeSelected: (MonitoringMode) -> Unit,
    enabled: Boolean
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = Color(0xFFE8EEF5),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Phone Mode Option
            val isPhoneSelected = selectedMode == MonitoringMode.PHONE
            Surface(
                onClick = { if (enabled) onModeSelected(MonitoringMode.PHONE) },
                shape = RoundedCornerShape(8.dp),
                color = if (isPhoneSelected) Color.White else Color.Transparent,
                shadowElevation = if (isPhoneSelected) 2.dp else 0.dp,
                modifier = Modifier.weight(1f)
            ) {
                Column(
                    modifier = Modifier.padding(vertical = 10.dp, horizontal = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "Phone 模式",
                        fontWeight = if (isPhoneSelected) FontWeight.Bold else FontWeight.Medium,
                        fontSize = 14.sp,
                        color = if (isPhoneSelected) Color(0xFF1565C0) else Color(0xFF555555)
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "吸氣呵氣檢測",
                        fontSize = 11.sp,
                        fontWeight = if (isPhoneSelected) FontWeight.Medium else FontWeight.Normal,
                        color = if (isPhoneSelected) Color(0xFF1E88E5) else Color(0xFF757575)
                    )
                }
            }

            // Stetho Mode Option
            val isStethoSelected = selectedMode == MonitoringMode.STETHO
            Surface(
                onClick = { if (enabled) onModeSelected(MonitoringMode.STETHO) },
                shape = RoundedCornerShape(8.dp),
                color = if (isStethoSelected) Color.White else Color.Transparent,
                shadowElevation = if (isStethoSelected) 2.dp else 0.dp,
                modifier = Modifier.weight(1f)
            ) {
                Column(
                    modifier = Modifier.padding(vertical = 10.dp, horizontal = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "Stetho 模式",
                        fontWeight = if (isStethoSelected) FontWeight.Bold else FontWeight.Medium,
                        fontSize = 14.sp,
                        color = if (isStethoSelected) Color(0xFF1565C0) else Color(0xFF555555)
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "麥克風貼胸聽診",
                        fontSize = 11.sp,
                        fontWeight = if (isStethoSelected) FontWeight.Medium else FontWeight.Normal,
                        color = if (isStethoSelected) Color(0xFF1E88E5) else Color(0xFF757575)
                    )
                }
            }
        }
    }
}

@Composable
fun ProbabilityMeters(wheezeProb: Float, coughProb: Float, mode: MonitoringMode) {
    val interferenceLabel = if (mode == MonitoringMode.STETHO) {
        "體音與接觸雜訊 (Body Noise)"
    } else {
        "干擾數值 (Interference)"
    }

    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Wheeze Meter
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Wheeze (喘鳴率)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Text("${(wheezeProb * 100).toInt()}%", color = Color(0xFFE65100), fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
                Spacer(modifier = Modifier.height(4.dp))
                LinearProgressIndicator(
                    progress = { wheezeProb },
                    modifier = Modifier.fillMaxWidth().height(8.dp),
                    color = Color(0xFFFF9800),
                    trackColor = Color(0xFFFFE0B2)
                )
            }

            // Interference Meter
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(interferenceLabel, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Text("${(coughProb * 100).toInt()}%", color = Color(0xFF0288D1), fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
                Spacer(modifier = Modifier.height(4.dp))
                LinearProgressIndicator(
                    progress = { coughProb },
                    modifier = Modifier.fillMaxWidth().height(8.dp),
                    color = Color(0xFF03A9F4),
                    trackColor = Color(0xFFB3E5FC)
                )
            }
        }
    }
}

@Composable
fun ActionButtons(
    state: RecordingState,
    onToggleRecording: () -> Unit,
    onSonarCalibrate: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Button(
            onClick = onToggleRecording,
            enabled = state != RecordingState.ANALYZING && state != RecordingState.CALIBRATING,
            modifier = Modifier.weight(1f),
            colors = ButtonDefaults.buttonColors(
                containerColor = when (state) {
                    RecordingState.RECORDING -> Color(0xFFE53935)
                    RecordingState.ANALYZING -> Color(0xFFFFA000)
                    else -> Color(0xFF43A047)
                }
            )
        ) {
            when (state) {
                RecordingState.RECORDING -> Text("停止分析", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                RecordingState.ANALYZING -> Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = Color.White,
                        strokeWidth = 2.dp
                    )
                    Text("AI 分析中...", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                }
                else -> Text("開始檢測", fontSize = 15.sp, fontWeight = FontWeight.Bold)
            }
        }

        OutlinedButton(
            onClick = onSonarCalibrate,
            enabled = state != RecordingState.RECORDING && state != RecordingState.ANALYZING && state != RecordingState.CALIBRATING,
            modifier = Modifier.weight(1f)
        ) {
            Text("聲納校正", fontSize = 15.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun HistoryList(
    records: List<AnalysisRecord>,
    playingPath: String?,
    onPlay: (String) -> Unit,
    onShare: (AnalysisRecord) -> Unit,
    onDelete: (AnalysisRecord) -> Unit
) {
    val sdf = remember { SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault()) }

    if (records.isEmpty()) {
        Text("尚無歷史檢測紀錄", color = Color.Gray, fontSize = 14.sp)
    } else {
        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            records.take(10).forEach { record ->
                val isPlaying = playingPath != null && playingPath == record.audioFilePath
                var showDeleteItemDialog by remember(record.id) { mutableStateOf(false) }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(sdf.format(Date(record.timestamp)), fontSize = 12.sp, color = Color.Gray)
                                Text(
                                    "喘鳴: ${(record.wheezeProb * 100).toInt()}% | 干擾: ${(record.coughProb * 100).toInt()}%",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium
                                )
                            }

                            Text(
                                text = if (record.triggeredAlarm) "警報" else "正常",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = if (record.triggeredAlarm) Color.Red else Color(0xFF2E7D32)
                            )
                        }

                        Spacer(modifier = Modifier.height(4.dp))

                        // Audio Control Row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            if (!record.audioFilePath.isNullOrEmpty()) {
                                TextButton(
                                    onClick = { onPlay(record.audioFilePath) },
                                    contentPadding = PaddingValues(horizontal = 8.dp)
                                ) {
                                    Text(if (isPlaying) "暫停" else "播放", fontSize = 13.sp)
                                }

                                TextButton(
                                    onClick = { onShare(record) },
                                    contentPadding = PaddingValues(horizontal = 8.dp)
                                ) {
                                    Text("分享", fontSize = 13.sp)
                                }
                            }

                            TextButton(
                                onClick = { showDeleteItemDialog = true },
                                contentPadding = PaddingValues(horizontal = 8.dp)
                            ) {
                                Text("刪除", fontSize = 13.sp, color = Color.Red)
                            }
                        }
                    }

                    if (showDeleteItemDialog) {
                        AlertDialog(
                            onDismissRequest = { showDeleteItemDialog = false },
                            title = { Text("確認刪除此筆紀錄？", fontWeight = FontWeight.Bold) },
                            text = { Text("將會永久刪除此筆檢測數據以及對應的錄音檔案。") },
                            confirmButton = {
                                TextButton(
                                    onClick = {
                                        showDeleteItemDialog = false
                                        onDelete(record)
                                    }
                                ) {
                                    Text("確認刪除", color = Color(0xFFD32F2F), fontWeight = FontWeight.Bold)
                                }
                            },
                            dismissButton = {
                                TextButton(onClick = { showDeleteItemDialog = false }) {
                                    Text("取消")
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun SpectrogramSqueakPreview() {
    MaterialTheme {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Option preview
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "即時 Mel-Spectrogram (Fire 視圖)",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.Gray
                )

                Surface(
                    color = Color(0xFFFFF8E1),
                    shape = RoundedCornerShape(16.dp),
                    border = BorderStroke(1.dp, Color(0xFFFFB300))
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            "短促過渡音 (Squeak)",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFFE65100),
                            maxLines = 1
                        )
                    }
                }
            }

            Box(modifier = Modifier.fillMaxWidth()) {
                MelSpectrogramView(
                    melData = null,
                    modifier = Modifier.fillMaxWidth()
                )

                Surface(
                    color = Color.Black.copy(alpha = 0.75f),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, Color(0xFFFFC107)),
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            "捕捉到極短促過渡音 (Squeak)",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFFFFD54F),
                            maxLines = 1
                        )
                    }
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ModeSelectorPreview() {
    MaterialTheme {
        Column(modifier = Modifier.padding(16.dp)) {
            ModeSelector(
                selectedMode = MonitoringMode.PHONE,
                onModeSelected = {},
                enabled = true
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun HistorySectionPreview() {
    MaterialTheme {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "最近檢測紀錄",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )

                TextButton(onClick = {}) {
                    Text(
                        "清除全部",
                        color = Color(0xFFD32F2F),
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }
    }
}

