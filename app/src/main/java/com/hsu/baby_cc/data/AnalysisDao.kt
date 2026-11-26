package com.hsu.baby_cc.data

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query

@Dao
interface AnalysisDao {
    @Insert
    suspend fun insert(record: AnalysisRecord): Long

    @Delete
    suspend fun delete(record: AnalysisRecord)

    @Query("DELETE FROM analysis_records")
    suspend fun deleteAll()

    @Query("SELECT * FROM analysis_records ORDER BY timestamp DESC")
    suspend fun getAllRecords(): List<AnalysisRecord>
}
