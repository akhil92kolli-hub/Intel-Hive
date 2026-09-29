package com.intellihive.worker.inference

import com.intellihive.worker.service.workerProtocolGson
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ShardInferenceTest {
    @Test
    fun shardMustMatchModelAndUseInclusiveValidLayerRange() {
        val model = model()
        assertNull(shard().validate(model))
        assertEquals(
            "shard layer range is outside the model",
            shard(startLayer = 10, endLayer = 30).validate(model)
        )
        assertEquals(
            "shard version does not match loaded model",
            shard(version = "v2").validate(model)
        )
    }

    @Test
    fun activationValidatesIdentityShapeAndPayloadChecksum() {
        val activation = Activation.create(
            jobId = "job-1",
            requestId = "request-1",
            sequenceId = "sequence-1",
            modelId = "model-1",
            modelVersion = "v1",
            sourceWorker = "worker-a",
            destinationWorker = "worker-b",
            layer = 9,
            dtype = "uint8",
            shape = listOf(3),
            payload = byteArrayOf(1, 2, 3)
        )

        assertNull(activation.validate())
        assertEquals(71, activation.checksum.length)
        assertEquals("activation checksum mismatch", activation.copy(payload = byteArrayOf(3, 2, 1)).validate())
        assertEquals(
            "activation shape must contain positive dimensions",
            activation.copy(shape = listOf(1, 0, 3072)).validate()
        )
    }

    @Test
    fun stepsRequireOrderedPrefillAndSingleTokenDecode() {
        assertNull(step(InferencePhase.PREFILL, 0, ShardInput.Prompt("hello")).validate())
        assertNull(step(InferencePhase.DECODE, 1, ShardInput.TokenIds(listOf(42))).validate())
        assertEquals(
            "prefill position must be zero",
            step(InferencePhase.PREFILL, 1, ShardInput.Prompt("hello")).validate()
        )
        assertEquals(
            "decode input must contain exactly one token",
            step(InferencePhase.DECODE, 1, ShardInput.TokenIds(listOf(42, 43))).validate()
        )
    }

    @Test
    fun activationConstructorDefensivelyCopiesPayload() {
        val payload = byteArrayOf(1, 2, 3)
        val activation = Activation.create(
            jobId = "job-1",
            requestId = "request-1",
            sequenceId = "sequence-1",
            modelId = "model-1",
            modelVersion = "v1",
            sourceWorker = "worker-a",
            destinationWorker = "worker-b",
            layer = 9,
            dtype = "uint8",
            shape = listOf(3),
            payload = payload
        )
        payload[0] = 0
        assertTrue(activation.validate() == null)
        assertTrue(activation.payload.contentEquals(byteArrayOf(1, 2, 3)))
    }

    @Test
    fun activationJsonUsesSharedSnakeCaseAndBase64PayloadContract() {
        val activation = Activation.create(
            jobId = "job-1",
            requestId = "request-1",
            sequenceId = "job-1:attempt-0",
            modelId = "model-1",
            modelVersion = "v1",
            sourceWorker = "worker-a",
            destinationWorker = "worker-b",
            layer = 9,
            dtype = "uint8",
            shape = listOf(3),
            payload = byteArrayOf(1, 2, 3)
        )

        val gson = workerProtocolGson
        val json = gson.toJson(activation)
        assertTrue(json.contains("\"sequence_id\":\"job-1:attempt-0\""))
        assertTrue(json.contains("\"payload\":\"AQID\""))
        assertTrue(json.contains("\"checksum\":\"sha256:"))
        assertNull(gson.fromJson(json, Activation::class.java).validate())
    }

    private fun model() = ModelSpec(
        modelId = "model-1",
        version = "v1",
        architecture = "llama",
        layerCount = 30,
        modelPath = "/models/model.gguf"
    )

    private fun shard(
        startLayer: Int = 0,
        endLayer: Int = 9,
        version: String = "v1"
    ) = ShardSpec(
        modelId = "model-1",
        version = version,
        startLayer = startLayer,
        endLayer = endLayer,
        memoryBytes = 1024,
        inputSpec = TensorSpec(listOf(1, 128, 3072), "float32"),
        outputSpec = TensorSpec(listOf(1, 128, 3072), "float32")
    )

    private fun step(phase: InferencePhase, position: Int, input: ShardInput) = InferenceStep(
        jobId = "job-1",
        requestId = "request-1",
        sequenceId = "sequence-1",
        phase = phase,
        position = position,
        input = input
    )
}
