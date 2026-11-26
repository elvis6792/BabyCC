package com.hsu.baby_cc.data

import android.content.Context
import android.content.SharedPreferences

class ProfileRepository(context: Context) {

    private val prefs: SharedPreferences = context.getSharedPreferences("user_profile_prefs", Context.MODE_PRIVATE)

    fun saveBaselineEnergy(baseline: FloatArray) {
        val str = baseline.joinToString(",")
        prefs.edit().putString("patient_baseline_energy", str).apply()
    }

    fun getBaselineEnergy(): FloatArray? {
        val str = prefs.getString("patient_baseline_energy", null) ?: return null
        return str.split(",").map { it.toFloat() }.toFloatArray()
    }

    fun setPatientName(name: String) {
        prefs.edit().putString("patient_name", name).apply()
    }

    fun getPatientName(): String {
        return prefs.getString("patient_name", "Baby Patient") ?: "Baby Patient"
    }
}
