create table if not exists public.benchmark_results (
  id uuid primary key default gen_random_uuid(),
  client_run_id uuid not null unique,
  created_at timestamptz not null default now(),
  platform text not null check (platform in ('android', 'ios')),
  worker_id text not null check (char_length(worker_id) between 1 and 128),
  app_version text not null,
  model_id text not null,
  model_version text not null,
  model_artifact_digest text not null
    check (model_artifact_digest ~ '^sha256:[0-9a-f]{64}$'),
  device_model text not null,
  os_version text not null,
  backend text not null,
  prefill_tokens integer not null check (prefill_tokens > 0),
  generated_tokens integer not null check (generated_tokens >= 0),
  total_time_ms bigint not null check (total_time_ms >= 0),
  tokens_per_second double precision not null check (tokens_per_second >= 0),
  prefill_tokens_per_second double precision not null
    check (prefill_tokens_per_second >= 0),
  generation_tokens_per_second double precision not null
    check (generation_tokens_per_second >= 0),
  peak_rss_mb bigint check (peak_rss_mb >= 0),
  peak_gpu_mb bigint check (peak_gpu_mb >= 0),
  initial_temperature_c real,
  peak_temperature_c real,
  final_temperature_c real,
  activation_size_bytes bigint check (activation_size_bytes >= 0),
  activation_dtype text,
  activation_shape jsonb,
  raw_result jsonb not null check (jsonb_typeof(raw_result) = 'object')
);

create index if not exists benchmark_results_created_at_idx
  on public.benchmark_results (created_at desc);
create index if not exists benchmark_results_model_platform_idx
  on public.benchmark_results (model_id, platform, created_at desc);

alter table public.benchmark_results enable row level security;

revoke all on table public.benchmark_results from anon, authenticated;
grant insert on table public.benchmark_results to anon, authenticated;

drop policy if exists "worker clients can submit benchmark results"
  on public.benchmark_results;
create policy "worker clients can submit benchmark results"
  on public.benchmark_results
  for insert
  to anon, authenticated
  with check (
    model_id = 'qwen2.5-3b-instruct'
    and model_version = '1.0.0'
    and model_artifact_digest =
      'sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'
  );

insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values (
  'model-artifacts',
  'model-artifacts',
  true,
  null,
  array['application/octet-stream', 'application/json']::text[]
)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

comment on table public.benchmark_results is
  'Append-only mobile worker benchmark telemetry. Anonymous clients may insert but cannot read, update, or delete.';

-- Reserved public object path after the project Storage limit is raised:
-- model-artifacts/qwen2.5-3b-instruct/1.0.0/qwen2.5-3b-instruct-q4_k_m.gguf
