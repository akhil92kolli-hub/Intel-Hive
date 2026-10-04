package com.intellihive.worker.benchmark

import com.intellihive.worker.inference.ShardPlacementProfile
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BenchmarkSuitePlanTest {
    @Test
    fun quickRunsOneComparableTokenTargetForEveryBackend() {
        val steps = BenchmarkSuitePlan.stepsFor(CharacterizationMode.QUICK)
        assertEquals(4, steps.size)
        assertEquals(
            listOf(
                BenchmarkStepKind.LLAMA_CPP_BASELINE,
                BenchmarkStepKind.CPU_ONLY,
                BenchmarkStepKind.VULKAN_ONLY,
                BenchmarkStepKind.HYBRID
            ),
            steps.map { it.kind }
        )
        assertTrue(steps.all {
            it.generatedTokens == 32 &&
                it.graphThreads == 4 &&
                it.measurementPhase == "comparison"
        })
    }

    @Test
    fun fullRuns32And64And128ForBaselineAndAllShardBackends() {
        val steps = BenchmarkSuitePlan.stepsFor(CharacterizationMode.FULL)
        assertEquals(12, steps.size)
        assertEquals(
            listOf(32, 64, 128),
            steps.filter { it.kind == BenchmarkStepKind.LLAMA_CPP_BASELINE }
                .map { it.generatedTokens }
        )
        assertEquals(
            listOf(32, 64, 128),
            steps.filter { it.kind == BenchmarkStepKind.CPU_ONLY }
                .map { it.generatedTokens }
        )
        assertEquals(
            listOf(32, 64, 128),
            steps.filter { it.kind == BenchmarkStepKind.VULKAN_ONLY }
                .map { it.generatedTokens }
        )
        assertEquals(
            listOf(32, 64, 128),
            steps.filter { it.kind == BenchmarkStepKind.HYBRID }
                .map { it.generatedTokens }
        )
        assertEquals(
            ShardPlacementProfile.CPU_ONLY,
            steps.first { it.kind == BenchmarkStepKind.LLAMA_CPP_BASELINE }.profile
        )
        assertEquals(
            ShardPlacementProfile.VULKAN_ALL,
            steps.first { it.kind == BenchmarkStepKind.VULKAN_ONLY }.profile
        )
        assertEquals(
            ShardPlacementProfile.VULKAN_S0_S1,
            steps.first { it.kind == BenchmarkStepKind.HYBRID }.profile
        )
        assertTrue(steps.all { it.durationMs == 300_000L })
    }

    @Test
    fun resumeSelectsPendingAndCancelledButNotFailedSteps() {
        val initial = BenchmarkSuiteCheckpoint(mode = CharacterizationMode.QUICK)
        val steps = initial.steps.mapIndexed { index, step ->
            when (index) {
                0 -> step.copy(attempts = listOf(BenchmarkStepAttempt(status = BenchmarkStepStatus.COMPLETED)))
                1 -> step.copy(attempts = listOf(BenchmarkStepAttempt(status = BenchmarkStepStatus.FAILED)))
                2 -> step.copy(attempts = listOf(BenchmarkStepAttempt(status = BenchmarkStepStatus.CANCELLED)))
                else -> step
            }
        }
        assertEquals(
            listOf(initial.steps[2].definition.id) + initial.steps.drop(3).map { it.definition.id },
            BenchmarkSuitePlan.resumableSteps(initial.copy(steps = steps)).map { it.definition.id }
        )
    }

    @Test
    fun interruptedRunningAttemptIsPersistedAsCancelledOnRecovery() {
        val directory = createTempDir(prefix = "benchmark-checkpoint-test")
        try {
            val store = BenchmarkSuiteCheckpointStore(File(directory, "checkpoint.json"))
            val checkpoint = BenchmarkSuiteCheckpoint(mode = CharacterizationMode.QUICK)
            val running = checkpoint.steps.first().copy(
                attempts = listOf(BenchmarkStepAttempt(status = BenchmarkStepStatus.RUNNING))
            )
            store.write(checkpoint.copy(steps = listOf(running) + checkpoint.steps.drop(1)))
            val recovered = store.recoverInterrupted(CharacterizationMode.QUICK)
            assertEquals(BenchmarkStepStatus.CANCELLED, recovered.steps.first().latestAttempt?.status)
            assertTrue(recovered.steps.first().latestAttempt?.errorReason.orEmpty().contains("process stopped"))
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun resetCreatesNewRunInRequestedMode() {
        val directory = createTempDir(prefix = "benchmark-checkpoint-reset-test")
        try {
            val store = BenchmarkSuiteCheckpointStore(File(directory, "checkpoint.json"))
            val original = BenchmarkSuiteCheckpoint(mode = CharacterizationMode.QUICK)
            store.write(original)
            val reset = store.reset(CharacterizationMode.FULL)
            assertFalse(reset.suiteId == original.suiteId)
            assertEquals(CharacterizationMode.FULL, reset.mode)
            assertEquals(12, reset.steps.size)
        } finally {
            directory.deleteRecursively()
        }
    }
}
