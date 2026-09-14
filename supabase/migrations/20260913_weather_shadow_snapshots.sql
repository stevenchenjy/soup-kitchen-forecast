-- Separate, immutable prospective experiment log. Run this migration manually
-- before enabling shared weather-shadow capture; no existing table is changed.
create table if not exists public.weather_shadow_runs (
  run_id uuid primary key,
  location_id text not null,
  service_date date not null,
  cutoff_at timestamptz not null,
  weather_retrieved_at timestamptz,
  recorded_at timestamptz not null,
  persisted_at timestamptz not null default clock_timestamp(),
  status text not null check (
    status in ('paired', 'weather_unavailable', 'candidate_unavailable', 'cutoff_missed')
  ),
  payload jsonb not null check (jsonb_typeof(payload) = 'object'),
  constraint weather_shadow_paired_before_cutoff check (
    status <> 'paired' or (
      weather_retrieved_at is not null
      and weather_retrieved_at <= cutoff_at
      and recorded_at <= cutoff_at
    )
  )
);

create index if not exists idx_weather_shadow_runs_location_date
  on public.weather_shadow_runs (location_id, service_date, recorded_at, run_id);

comment on table public.weather_shadow_runs is
  'Immutable paired F6/weather candidate forecasts with cutoff and raw forecast provenance. New UUID for each rerun; attendance outcomes are joined at evaluation time.';

comment on column public.weather_shadow_runs.persisted_at is
  'Database insertion receipt. Generated on the server and excluded from application INSERT privileges; evaluate against preparation cutoff.';

-- Staff-facing anonymous/authenticated clients cannot read or write this log.
-- The server capture process needs the service-role key. RLS bypass does not
-- bypass PostgreSQL privileges or the immutable-record trigger below.
alter table public.weather_shadow_runs enable row level security;
revoke all on public.weather_shadow_runs from public, anon, authenticated;
revoke all on public.weather_shadow_runs from service_role;
grant select on public.weather_shadow_runs to service_role;
-- Column-specific INSERT makes persisted_at server-owned. Sending an explicit
-- receipt through PostgREST is forbidden even with the service-role key.
grant insert (
  run_id, location_id, service_date, cutoff_at, weather_retrieved_at,
  recorded_at, status, payload
) on public.weather_shadow_runs to service_role;

create or replace function public.reject_weather_shadow_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog
as $$
begin
  raise exception 'weather shadow records are immutable'
    using errcode = '23000';
end;
$$;

-- A privileged maintenance operator must deliberately remove the trigger to
-- alter old snapshots. Ordinary application code has no update/delete path.
do $$
begin
  if not exists (
    select 1 from pg_trigger
    where tgname = 'weather_shadow_runs_immutable'
      and tgrelid = 'public.weather_shadow_runs'::regclass
  ) then
    create trigger weather_shadow_runs_immutable
      before update or delete or truncate on public.weather_shadow_runs
      for each statement execute function public.reject_weather_shadow_mutation();
  end if;
end;
$$;
