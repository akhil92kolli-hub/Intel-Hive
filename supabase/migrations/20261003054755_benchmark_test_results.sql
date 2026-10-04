alter table public.benchmark_results
  add column if not exists benchmark_version text,
  add column if not exists benchmark_mode text,
  add column if not exists selected_placement text,
  add column if not exists selected_cpu_threads integer;

alter table public.benchmark_results
  drop constraint if exists benchmark_results_benchmark_mode_check,
  drop constraint if exists benchmark_results_selected_placement_check,
  drop constraint if exists benchmark_results_selected_cpu_threads_check;

alter table public.benchmark_results
  add constraint benchmark_results_benchmark_mode_check
    check (benchmark_mode is null or benchmark_mode in ('quick', 'full')),
  add constraint benchmark_results_selected_placement_check
    check (selected_placement is null or selected_placement in (
      'cpu_only', 'vulkan_s0', 'vulkan_s0_s1', 'vulkan_all',
      'vulkan_s2', 'vulkan_s1_s2'
    )),
  add constraint benchmark_results_selected_cpu_threads_check
    check (selected_cpu_threads is null or selected_cpu_threads between 1 and 8);

create table if not exists public.benchmark_test_results (
  id uuid primary key default gen_random_uuid(),
  client_test_id uuid not null unique,
  suite_client_run_id uuid not null
    references public.benchmark_results(client_run_id) on delete cascade,
  created_at timestamptz not null default now(),
  platform text not null check (platform in ('android', 'ios')),
  worker_id text not null check (char_length(worker_id) between 1 and 128),
  model_id text not null,
  model_version text not null,
  model_artifact_digest text not null
    check (model_artifact_digest ~ '^sha256:[0-9a-f]{64}$'),
  profile text not null check (profile in (
    'llama_cpp_baseline', 'cpu_only', 'vulkan_s0', 'vulkan_s0_s1', 'vulkan_all',
    'vulkan_s2', 'vulkan_s1_s2'
  )),
  requested_backends jsonb not null check (jsonb_typeof(requested_backends) = 'object'),
  actual_backends jsonb not null check (jsonb_typeof(actual_backends) = 'object'),
  cpu_threads integer not null check (cpu_threads between 1 and 8),
  requested_gpu_layers integer not null default 0 check (requested_gpu_layers between 0 and 36),
  observed_gpu_layers integer not null default 0 check (observed_gpu_layers between 0 and 36),
  measurement_phase text not null check (measurement_phase in ('comparison', 'screening', 'sustained')),
  duration_ms bigint not null check (duration_ms >= 0),
  generated_tokens integer not null check (generated_tokens >= 0),
  throughput double precision not null check (throughput >= 0),
  latency_percentiles jsonb,
  memory jsonb not null check (jsonb_typeof(memory) = 'object'),
  thermal jsonb not null check (jsonb_typeof(thermal) = 'object'),
  backend_statistics jsonb not null check (jsonb_typeof(backend_statistics) = 'object'),
  status text not null check (status in ('COMPLETED', 'FAILED', 'SKIPPED', 'CANCELLED')),
  failure_reason text,
  raw_result jsonb not null check (jsonb_typeof(raw_result) = 'object')
);

create index if not exists benchmark_test_results_suite_idx
  on public.benchmark_test_results (suite_client_run_id, created_at);
create index if not exists benchmark_test_results_profile_idx
  on public.benchmark_test_results
    (model_artifact_digest, profile, measurement_phase, created_at desc);

alter table public.benchmark_test_results enable row level security;
revoke all on table public.benchmark_test_results from anon, authenticated;
grant insert on table public.benchmark_test_results to anon, authenticated;

drop policy if exists "worker clients can submit benchmark test results"
  on public.benchmark_test_results;
create policy "worker clients can submit benchmark test results"
  on public.benchmark_test_results
  for insert
  to anon, authenticated
  with check (
    model_id = 'qwen2.5-3b-instruct'
    and model_version = '1.0.0'
    and model_artifact_digest =
      'sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'
  );

comment on table public.benchmark_test_results is
  'Append-only per-profile measurements belonging to one benchmark_results suite.';
