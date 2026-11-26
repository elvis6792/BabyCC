package com.hsu.baby_cc.engine

enum class MonitoringMode {
    PHONE, // 15 seconds home monitoring
    STETHO // Stethoscope continuous monitoring
}

enum class AlertLevel {
    NORMAL,  // Normal
    WARNING, // Yellow Warning (Suspected mild / cough-associated wheezing)
    ALARM    // Red Alarm (Asthma Attack)
}

data class DecisionResult(
    val level: AlertLevel,
    val consecutiveHits: Int = 0,
    val targetHits: Int = 3,
    val maxWheezeProb: Float = 0.0f,
    val hasTransientSqueak: Boolean = false
)

class AcousticDecisionEngine(private val mode: MonitoringMode) {

    private val windowResults = mutableListOf<Pair<Float, Float>>() // <wheeze_prob, cough_prob>

    fun processWindow(wheezeProb: Float, coughProb: Float) {
        windowResults.add(Pair(wheezeProb, coughProb))
    }

    fun evaluate(): DecisionResult {
        if (windowResults.isEmpty()) return DecisionResult(AlertLevel.NORMAL)

        return when (mode) {
            MonitoringMode.PHONE -> evaluatePhoneMode()
            MonitoringMode.STETHO -> evaluateStethoMode()
        }
    }

    private fun evaluatePhoneMode(): DecisionResult {
        var consecutiveWheeze = 0
        var allowedBreaks = 0
        var maxConsecutiveHits = 0
        var maxWheezeProb = 0.0f
        var strongHits = 0
        var mildHits = 0

        for ((wheezeProb, coughProb) in windowResults) {
            if (wheezeProb > maxWheezeProb) maxWheezeProb = wheezeProb

            // 智慧咳嗽爆音/清喉嚨遮罩：當 AI 咳嗽頭識別出咳嗽暴衝與咽喉摩擦尾音 (Cough ≥ 0.15) 時，該視窗絕不誤計為喘鳴
            val isPureCough = coughProb >= 0.15f
            val effectiveWheeze = if (isPureCough) 0.0f else wheezeProb

            if (effectiveWheeze >= 0.20f) {
                consecutiveWheeze++
                allowedBreaks = 0
                if (consecutiveWheeze > maxConsecutiveHits) {
                    maxConsecutiveHits = consecutiveWheeze
                }
            } else {
                if (consecutiveWheeze > 0) {
                    allowedBreaks++
                    if (allowedBreaks > 2) {
                        consecutiveWheeze = 0
                        allowedBreaks = 0
                    }
                }
            }

            if (effectiveWheeze >= 0.55f) {
                strongHits++
            } else if (effectiveWheeze >= 0.20f) {
                mildHits++
            }
        }

        val isIsolatedTransientSqueak = (strongHits == 0 && mildHits == 1)

        val level = when {
            // 喘鳴發作長度門檻：必須為連續不間斷發作 (連續 ≥ 3 窗) 或強喘鳴 ≥ 2 窗，完全排除單秒咳嗽摩擦音
            maxConsecutiveHits >= 3 || strongHits >= 2 -> AlertLevel.ALARM
            // 預警：連續 ≥ 2 窗，或強喘鳴 ≥ 1 窗
            maxConsecutiveHits >= 2 || (strongHits + mildHits >= 3) -> AlertLevel.WARNING
            else -> AlertLevel.NORMAL
        }

        return DecisionResult(
            level = level,
            consecutiveHits = if (isIsolatedTransientSqueak) 0 else maxConsecutiveHits.coerceAtLeast(strongHits),
            targetHits = 3,
            maxWheezeProb = maxWheezeProb,
            hasTransientSqueak = isIsolatedTransientSqueak || (maxWheezeProb in 0.20f..0.54f)
        )
    }

    private fun evaluateStethoMode(): DecisionResult {
        var consecutiveWheeze = 0
        var allowedBreaks = 0
        var maxConsecutiveHits = 0
        var maxWheezeProb = 0.0f
        var strongHits = 0
        var mildHits = 0

        for ((wheezeProb, coughProb) in windowResults) {
            if (wheezeProb > maxWheezeProb) maxWheezeProb = wheezeProb

            val isPureCough = coughProb >= 0.15f
            val effectiveWheeze = if (isPureCough) 0.0f else wheezeProb

            if (effectiveWheeze >= 0.20f) {
                consecutiveWheeze++
                allowedBreaks = 0
                if (consecutiveWheeze > maxConsecutiveHits) {
                    maxConsecutiveHits = consecutiveWheeze
                }
            } else {
                if (consecutiveWheeze > 0) {
                    allowedBreaks++
                    if (allowedBreaks > 2) {
                        consecutiveWheeze = 0
                        allowedBreaks = 0
                    }
                }
            }

            if (effectiveWheeze >= 0.55f) {
                strongHits++
            } else if (effectiveWheeze >= 0.20f) {
                mildHits++
            }
        }

        val level = when {
            maxConsecutiveHits >= 3 || strongHits >= 2 || (strongHits + mildHits >= 3) -> AlertLevel.ALARM
            maxConsecutiveHits >= 2 || strongHits >= 1 || mildHits >= 2 || maxWheezeProb >= 0.55f -> AlertLevel.WARNING
            else -> AlertLevel.NORMAL
        }

        return DecisionResult(
            level = level,
            consecutiveHits = maxConsecutiveHits.coerceAtLeast(strongHits + mildHits),
            targetHits = 3,
            maxWheezeProb = maxWheezeProb,
            hasTransientSqueak = false
        )
    }

    fun reset() {
        windowResults.clear()
    }
}
