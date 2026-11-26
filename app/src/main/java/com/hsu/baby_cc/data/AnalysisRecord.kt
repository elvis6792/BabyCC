package com.hsu.baby_cc.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "analysis_records")
data class AnalysisRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val wheezeProb: Float,
    val coughProb: Float,
    val triggeredAlarm: Boolean,
    val audioFilePath: String? = null
)
