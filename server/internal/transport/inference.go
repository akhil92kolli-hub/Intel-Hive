package transport

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/scheduler"
	"github.com/google/uuid"
)

const maxInferenceRequestBytes = 1 << 20

type InferenceHandler struct {
	Scheduler *scheduler.Scheduler
	Workers   *WorkerSocketHandler
}

type inferenceRequest struct {
	JobID     string `json:"job_id,omitempty"`
	ModelID   string `json:"model_id"`
	Prompt    string `json:"prompt"`
	MaxTokens *int   `json:"max_tokens,omitempty"`
}

type inferenceResponse struct {
	JobID             string   `json:"job_id"`
	ModelID           string   `json:"model_id"`
	Output            string   `json:"output"`
	GeneratedTokenIDs []uint32 `json:"generated_token_ids"`
	ExecutionTimeMs   int64    `json:"execution_time_ms"`
	TokensPerSec      float64  `json:"tokens_per_second"`
}

func (h *InferenceHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if h.Scheduler == nil || h.Workers == nil {
		http.Error(w, "inference service is not configured", http.StatusServiceUnavailable)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxInferenceRequestBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	var request inferenceRequest
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		http.Error(w, "request body must contain one JSON object", http.StatusBadRequest)
		return
	}
	request.ModelID = strings.TrimSpace(request.ModelID)
	if request.ModelID == "" || strings.TrimSpace(request.Prompt) == "" {
		http.Error(w, "model_id and prompt are required", http.StatusBadRequest)
		return
	}
	if request.JobID == "" {
		request.JobID = uuid.NewString()
	}
	maxTokens := 32
	if request.MaxTokens != nil {
		maxTokens = *request.MaxTokens
	}
	if maxTokens < 1 || maxTokens > 256 {
		http.Error(w, "max_tokens must be between 1 and 256", http.StatusBadRequest)
		return
	}

	layerCount, supported := supportedModelLayerCount(request.ModelID)
	if !supported {
		http.Error(w, "model is not in the Phase-0 model catalog", http.StatusNotFound)
		return
	}
	shards, found := h.Workers.ModelShards(request.ModelID)
	if !found {
		http.Error(w, "no workers have advertised shards for this model", http.StatusServiceUnavailable)
		return
	}
	if err := model.ValidatePartition(layerCount, layerRanges(shards)); err != nil {
		http.Error(w, fmt.Sprintf("registered shard catalog is incomplete: %v", err), http.StatusServiceUnavailable)
		return
	}

	plan, err := h.Scheduler.BuildPlan(request.JobID, request.ModelID, shards)
	if err != nil {
		http.Error(w, "no complete worker pipeline is available: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	result, err := h.Scheduler.Generate(r.Context(), plan, request.Prompt, maxTokens)
	if err != nil {
		http.Error(w, "inference failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(inferenceResponse{
		JobID: result.JobID, ModelID: request.ModelID, Output: string(result.Text),
		GeneratedTokenIDs: result.TokenIDs,
		ExecutionTimeMs:   result.ExecutionTimeMs, TokensPerSec: result.TokensPerSec,
	}); err != nil {
		return
	}
}

func layerRanges(shards []model.ModelShard) []model.LayerRange {
	ranges := make([]model.LayerRange, len(shards))
	for i, shard := range shards {
		ranges[i] = shard.Layers
	}
	return ranges
}

func supportedModelLayerCount(modelID string) (int, bool) {
	switch modelID {
	case "qwen2.5-3b-instruct":
		return 36, true
	case "qwen2.5-3b":
		return 36, true
	default:
		return 0, false
	}
}
