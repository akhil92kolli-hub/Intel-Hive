alter table public.benchmark_results
  add column if not exists execution_mode text;

update public.benchmark_results
set execution_mode = 'single_device_all_shards'
where execution_mode is null;

alter table public.benchmark_results
  alter column execution_mode set default 'single_device_all_shards',
  alter column execution_mode set not null;

alter table public.benchmark_results
  drop constraint if exists benchmark_results_execution_mode_check;

alter table public.benchmark_results
  add constraint benchmark_results_execution_mode_check
  check (execution_mode in ('single_device_all_shards', 'distributed_pipeline'));

comment on column public.benchmark_results.execution_mode is
  'Whether one device executed every logical shard or independent workers formed a distributed pipeline.';
