package com.intellihive.worker.service

import com.intellihive.worker.inference.Activation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WorkerAssignmentTest {
    @Test
    fun parsesDecodeAssignmentAndBase64Activation() {
        val json = """
            {
              "request_id":"job-1", "model_version":"v1", "worker_id":"worker-b", "previous_worker":"worker-a", "next_worker":"worker-c",
              "assignment_id":"assignment-1",
              "job_id":"job-1",
              "model_id":"qwen2.5-3b-instruct",
              "shard_id":"shard-10-19",
              "phase":"DECODE",
              "sequence_id":"job-1:0",
              "position":1,
              "activation":${workerProtocolGson.toJson(testActivation())},
              "layer_start":10,
              "layer_end":19,
              "sequence":1
            }
        """.trimIndent()

        val assignment = workerProtocolGson.fromJson(json, WorkerJobAssignment::class.java)

        assertEquals("assignment-1", assignment.assignmentId)
        assertEquals(WorkerJobAssignment.DECODE, assignment.phase)
        assertEquals(listOf<Byte>(1, 2), assignment.activation?.payload?.toList())
        assertNull(assignment.validate())
    }

    @Test
    fun validatesExactlyOneInputFormAndDecodeTokenCount() {
        val assignment = WorkerJobAssignment(
            requestId = "job-1", modelVersion = "v1", workerId = "worker-a", nextWorker = "worker-b",
            assignmentId = "assignment-1",
            jobId = "job-1",
            modelId = "model",
            shardId = "shard-0",
            phase = WorkerJobAssignment.DECODE,
            sequenceId = "job-1:0",
            position = 1,
            inputTokenIds = listOf(1, 2),
            layerStart = 0,
            layerEnd = 2,
            sequence = 1
        )

        assertEquals("decode steps must contain exactly one input token", assignment.validate())
    }

    private fun testActivation() = Activation.create("job-1", "job-1", "job-1:0", "qwen2.5-3b-instruct", "v1", "worker-a", "worker-b", 9, "uint8", listOf(2), byteArrayOf(1, 2)).copy(position = 1)

    @Test
    fun serializesCompletionWithWorkerProtocolFieldNames() {
        val json = workerProtocolGson.toJsonTree(
            WorkerJobComplete(
                assignmentId = "assignment-1",
                jobId = "job-1",
                workerId = "worker-1",
                sequenceId = "job-1:0",
                position = 1,
                activation = testActivation()
            )
        ).asJsonObject

        assertEquals("assignment-1", json.get("assignment_id").asString)
        assertEquals("job-1:0", json.get("sequence_id").asString)
        assertEquals(1, json.get("position").asInt)
        assertEquals("AQI=", json.getAsJsonObject("activation").get("payload").asString)
    }
}
