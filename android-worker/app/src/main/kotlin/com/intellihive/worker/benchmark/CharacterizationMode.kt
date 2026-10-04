package com.intellihive.worker.benchmark

import java.util.Locale

enum class CharacterizationMode(val wireValue: String) {
    QUICK("quick"),
    FULL("full");

    companion object {
        val cpuThreadConfigurations = listOf(2, 4, 6, 8)

        fun fromWireValue(value: String?): CharacterizationMode =
            entries.firstOrNull { it.wireValue == value?.lowercase(Locale.ROOT) } ?: QUICK
    }
}
