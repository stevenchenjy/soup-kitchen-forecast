-- Keep model_retrain_state synchronized even when an application-side
-- best-effort marker fails. Apply this migration to the same Supabase project
-- used by Streamlit and GitHub Actions.

create table if not exists public.model_retrain_state (
    location_id text primary key,
    dirty boolean not null default false,
    last_attendance_updated_at timestamptz,
    last_successful_training_at timestamptz,
    updated_at timestamptz not null default now()
);

create or replace function public.queue_model_retraining_from_attendance()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
    affected_location_id text;
    attendance_changed_at timestamptz;
begin
    affected_location_id := coalesce(new.location_id, old.location_id);
    attendance_changed_at := case
        when tg_op = 'DELETE' then now()
        else coalesce(new.updated_at, now())
    end;

    insert into public.model_retrain_state (
        location_id,
        dirty,
        last_attendance_updated_at,
        updated_at
    )
    values (
        affected_location_id,
        true,
        attendance_changed_at,
        attendance_changed_at
    )
    on conflict (location_id) do update
    set dirty = true,
        last_attendance_updated_at = excluded.last_attendance_updated_at,
        updated_at = excluded.updated_at;

    if tg_op = 'DELETE' then
        return old;
    end if;
    return new;
end;
$$;

revoke execute on function public.queue_model_retraining_from_attendance()
from public;

drop trigger if exists attendance_queue_model_retraining on public.attendance;

create trigger attendance_queue_model_retraining
after insert or update or delete on public.attendance
for each row
execute function public.queue_model_retraining_from_attendance();
