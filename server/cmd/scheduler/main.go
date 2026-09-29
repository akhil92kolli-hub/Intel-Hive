package main

import (
	"log"
	"net/http"
	"os"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/scheduler"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/registry"
	workertransport "github.com/akhil92kolli-hub/Intel-Hive/server/internal/transport"
)

func main() {
	logger := log.New(os.Stdout, "intel-hive-scheduler ", log.LstdFlags|log.LUTC)
	workers := registry.NewWorkerRegistry()
	executionScheduler := scheduler.New(nil)
	workerSocket := workertransport.NewWorkerSocketHandler(workers, logger)
	workerSocket.SetScheduler(executionScheduler)
	mux := http.NewServeMux()
	mux.Handle("/worker", workerSocket)
	mux.Handle("/inference", &workertransport.InferenceHandler{
		Scheduler: executionScheduler,
		Workers:   workerSocket,
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	address := os.Getenv("LISTEN_ADDR")
	if address == "" {
		address = ":8080"
	}
	logger.Printf("listening on %s", address)
	if err := http.ListenAndServe(address, mux); err != nil {
		logger.Fatalf("scheduler server stopped: %v", err)
	}
}
