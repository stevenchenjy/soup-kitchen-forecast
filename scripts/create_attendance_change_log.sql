create table if not exists public.attendance_change_log (
  id bigserial primary key,
  location_id text not null,
  service_date date not null,
  operation text not null,
  previous_visitors integer,
  new_visitors integer,
  changed_by text,
  created_at timestamptz not null default now()
);

comment on table public.attendance_change_log is
  'Stores the latest Staff-created attendance ADD receipt per user and location.';
comment on column public.attendance_change_log.previous_visitors is
  'Legacy undo field; unused by the Staff ADD receipt workflow.';
comment on column public.attendance_change_log.new_visitors is
  'Legacy undo field; unused by the Staff ADD receipt workflow.';

create index if not exists attendance_change_log_undo_idx
  on public.attendance_change_log (location_id, changed_by, created_at desc, id desc);
