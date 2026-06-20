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

create index if not exists attendance_change_log_undo_idx
  on public.attendance_change_log (location_id, changed_by, created_at desc, id desc);
