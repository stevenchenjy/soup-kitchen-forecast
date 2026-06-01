import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.auth import (
    authenticate_user,
    delete_user,
    get_user,
    hash_password,
    load_users,
    require_role,
    save_users,
    user_store_mode,
    validate_password,
    validate_user_record,
)
from src.config import DATE_COL, TARGET_COL, artifact_dir_for_location, model_file_for_location
from src.data_admin import delete_record, load_clean_data, save_clean_data, upsert_record
from src.location_config import save_locations, list_locations
from src.prediction_logs import (
    load_prediction_logs,
    prediction_log_store_mode,
    save_prediction_log,
    summarize_monitoring,
    update_prediction_logs_with_actual,
)
from src.predictor import VisitorPredictor

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Multi-Location Forecast Admin", layout="wide")



def login_gate() -> None:
    st.title("Login")
    st.caption("Please sign in to continue")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        if username != username.strip():
            st.warning("Username had leading or trailing spaces; trying the trimmed username.")
        user = authenticate_user(username.strip(), password)
        if user is None:
            st.error("Invalid username or password")
        else:
            st.session_state["user"] = {"username": user["username"]}
            st.rerun()
    st.stop()


def load_current_user() -> dict | None:
    stored_user = st.session_state.get("user", {})
    if not isinstance(stored_user, dict):
        return None
    username = str(stored_user.get("username", "")).strip()
    if not username:
        return None
    return get_user(username)


if "user" not in st.session_state:
    login_gate()

user = load_current_user()
if user is None:
    st.session_state.clear()
    st.error("Your account is no longer available. Please log in again.")
    login_gate()

if not require_role(user, {"master", "admin"}):
    st.error("Master/admin role required for this dashboard.")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.stop()

locations = list_locations()
loc_names = {loc.name: loc.id for loc in locations}

st.title("Multi-Location Forecast Admin")
st.caption("Each location has independent database, model artifacts, and prediction outputs")
if user_store_mode() == "local_json":
    st.warning(
        "User accounts are using local data/users.json fallback. On Streamlit Cloud this is not shared between "
        "separate admin and staff apps, so configure Supabase secrets before managing production accounts."
    )

sidebar = st.sidebar
sidebar.markdown(f"**User:** {user['username']} ({user['role']})")
if not locations:
    st.error("No locations are configured yet. Add a location before using the dashboard.")
    if sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.stop()

selected_name = sidebar.selectbox("Location", options=list(loc_names.keys()))
location_id = loc_names[selected_name]
if sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

model_path = model_file_for_location(location_id)
artifact_dir = artifact_dir_for_location(location_id)
predictor = VisitorPredictor(str(model_path)) if model_path.exists() else None



def render_prediction():
    st.subheader(f"Prediction - {selected_name}")
    if predictor is None:
        st.warning(f"Model not found for location '{location_id}'. Train this location first.")
        if st.button("Train this location", type="primary"):
            with st.spinner("Training..."):
                r = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "train_backtest.py"), "--location", location_id],
                    cwd=ROOT,
                )
            if r.returncode == 0:
                st.success("Training completed")
                st.rerun()
            else:
                st.error("Training failed")
        return

    c1, c2 = st.columns(2)
    with c1:
        buffer_pct = st.slider("Base meal buffer (%)", min_value=0, max_value=30, value=8, step=1)
    with c2:
        custom_date = st.text_input("Target service date (Sat/Sun, YYYY-MM-DD)", value="")

    if st.button("Generate prediction", type="primary"):
        try:
            pred = predictor.predict_next(target_date=custom_date or None, meal_buffer_pct=buffer_pct / 100.0)
            st.success(
                f"Location: {location_id} | Service Date: {pred.service_date:%Y-%m-%d} | "
                f"Segment: {pred.model_segment.upper()} | Point: {pred.predicted_visitors:.1f} | "
                f"Quantile: {pred.predicted_quantile:.1f} | Residual Buffer: +{pred.residual_buffer:.1f} | "
                f"Suggested Meals: {pred.suggested_meals}"
            )
            try:
                save_prediction_log(location_id, pred, created_by=user["username"], source_app="admin")
            except Exception:
                st.warning("Prediction was generated, but monitoring log could not be saved.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



def render_metrics():
    st.subheader(f"Backtest Metrics - {selected_name}")
    metrics_path = artifact_dir / "metrics.json"
    if not metrics_path.exists():
        st.info("No metrics available yet.")
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key, title in [("overall", "Overall"), ("sat", "Saturday"), ("sun", "Sunday")]:
        m = metrics.get(key, {})
        st.markdown(f"**{title}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{m.get('MAE', 0):.2f}")
        c2.metric("RMSE", f"{m.get('RMSE', 0):.2f}")
        c3.metric("MAPE", f"{m.get('MAPE', 0) * 100:.2f}%")
        c4.metric("P90AbsError", f"{m.get('P90AbsError', 0):.2f}")

    img1 = artifact_dir / "backtest_actual_vs_pred.png"
    img2 = artifact_dir / "backtest_abs_error.png"
    if img1.exists() and img2.exists():
        st.image([str(img1), str(img2)], caption=["Actual vs Pred/PredQ", "Absolute Error"])


def render_model_monitoring():
    st.subheader("Model Monitoring")
    scope = st.radio("Scope", options=["Selected location", "All locations"], horizontal=True)
    monitor_location_id = location_id if scope == "Selected location" else None
    st.caption(f"Prediction log store: {prediction_log_store_mode()}")

    try:
        summary = summarize_monitoring(location_id=monitor_location_id)
        logs = load_prediction_logs(location_id=monitor_location_id, limit=200)
    except Exception as exc:
        st.warning(f"Monitoring data could not be loaded: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    live_mae = summary.get("live_mae")
    c1.metric("Live MAE", "N/A" if live_mae is None else f"{live_mae:.2f}")
    c2.metric("Logs with actuals", int(summary.get("logs_with_actuals") or 0))
    c3.metric("Waste avoided", f"{summary.get('total_estimated_waste_avoided_meals', 0):.1f} meals")
    c4.metric("CO2e reduction", f"{summary.get('total_estimated_co2e_reduction_kg', 0):.1f} kg")

    if not logs:
        st.info("No prediction logs available yet.")
        return

    columns = [
        "service_date",
        "predicted_visitors",
        "suggested_meals",
        "actual_visitors",
        "absolute_error",
        "waste_avoided_meals",
        "estimated_co2e_reduction_kg",
        "prediction_created_at",
        "created_by",
        "source_app",
    ]
    if monitor_location_id is None:
        columns.insert(0, "location_id")

    rows = []
    for log in logs:
        row = {column: log.get(column) for column in columns}
        for column in [
            "predicted_visitors",
            "absolute_error",
            "waste_avoided_meals",
            "estimated_co2e_reduction_kg",
        ]:
            if row.get(column) is not None:
                row[column] = round(float(row[column]), 2)
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)



def render_data_ops():
    st.subheader(f"Data CRUD + Incremental Retraining - {selected_name}")
    df = load_clean_data(location_id)

    q1, q2 = st.columns(2)
    with q1:
        start = st.date_input("Start date", value=df[DATE_COL].min().date() if not df.empty else None)
    with q2:
        end = st.date_input("End date", value=df[DATE_COL].max().date() if not df.empty else None)

    if start and end and not df.empty:
        show = df[(df[DATE_COL].dt.date >= start) & (df[DATE_COL].dt.date <= end)].copy()
    else:
        show = df.copy()

    st.dataframe(show[[DATE_COL, TARGET_COL]], use_container_width=True, height=260)

    st.markdown("**Add / Update record**")
    a1, a2, a3 = st.columns([2, 2, 1])
    with a1:
        add_date = st.date_input("Service date", value=None, key="add_date")
    with a2:
        add_visitors = st.number_input("Visitors", min_value=0, max_value=10000, value=120, step=1)
    with a3:
        if st.button("Save record"):
            if add_date is None:
                st.error("Please select date")
            else:
                upsert_record(str(add_date), int(add_visitors), location_id)
                monitoring_updated = True
                try:
                    update_prediction_logs_with_actual(location_id, str(add_date), int(add_visitors))
                except Exception:
                    monitoring_updated = False
                    st.warning("Attendance was saved, but monitoring log could not be updated.")
                st.success("Saved")
                if monitoring_updated:
                    st.rerun()

    st.markdown("**Delete record**")
    del_options = [d.strftime("%Y-%m-%d") for d in df[DATE_COL].sort_values()] if not df.empty else []
    del_date = st.selectbox("Service date to delete", options=del_options, index=None, placeholder="Select date")
    if st.button("Delete selected record"):
        if del_date:
            delete_record(del_date, location_id)
            st.success(f"Deleted {del_date}")
            st.rerun()
        else:
            st.error("Please select date")

    st.markdown("**Bulk edit**")
    edit_df = st.data_editor(df[[DATE_COL, TARGET_COL]], num_rows="dynamic", use_container_width=True)
    if st.button("Save bulk edits"):
        save_clean_data(edit_df, location_id)
        st.success("Saved")
        st.rerun()

    if st.button("Incremental retraining", type="primary"):
        with st.spinner("Training..."):
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "retrain_incremental.py"), "--location", location_id],
                cwd=ROOT,
            )
        if r.returncode == 0:
            st.success("Training completed. Model updated.")
            st.rerun()
        else:
            st.error("Training failed.")


def render_master_accounts(users):
    st.subheader("Master/Admin Accounts")
    master_users = [account for account in users if require_role(account, {"master", "admin"})]
    if master_users:
        rows = [
            {
                "Username": account["username"],
                "Role": account["role"],
                "Authorized locations": ", ".join(account.get("authorized_locations", [])) or "None",
            }
            for account in master_users
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No master/admin accounts have been created yet.")

    st.markdown("**Create master/admin account**")
    with st.form("create_master_account"):
        new_admin_username = st.text_input("New admin username")
        new_admin_password = st.text_input("New admin password", type="password")
        create_admin_ok = st.form_submit_button("Create master/admin account")
    if create_admin_ok:
        new_admin_username = new_admin_username.strip()
        password_error = validate_password(new_admin_password)
        if not new_admin_username:
            st.error("Please enter a username.")
        elif password_error:
            st.error(password_error)
        elif any(account["username"].lower() == new_admin_username.lower() for account in users):
            st.error("That username already exists.")
        else:
            users.append(
                {
                    "username": new_admin_username,
                    "role": "master",
                    "authorized_locations": ["*"],
                    **hash_password(new_admin_password),
                }
            )
            save_users(users)
            st.success(f"Created master/admin account '{new_admin_username}'.")
            st.rerun()

    if not master_users:
        return

    st.markdown("**Reset master/admin password**")
    reset_admin = st.selectbox(
        "Master/admin account",
        options=[account["username"] for account in master_users],
        key="admin_password_reset_account",
    )
    with st.form("reset_admin_password"):
        reset_admin_password = st.text_input("New password", type="password")
        admin_password_has_outer_space = reset_admin_password != reset_admin_password.strip()
        if admin_password_has_outer_space:
            st.warning("This password begins or ends with whitespace.")
        confirm_admin_whitespace = st.checkbox(
            "Save admin password with leading/trailing whitespace",
            disabled=not admin_password_has_outer_space,
        )
        reset_admin_ok = st.form_submit_button("Reset password")
    if reset_admin_ok:
        password_error = validate_password(reset_admin_password)
        if password_error:
            st.error(password_error)
        elif admin_password_has_outer_space and not confirm_admin_whitespace:
            st.error("Please confirm before saving a password with leading or trailing whitespace.")
        else:
            for account in users:
                if account["username"] == reset_admin:
                    account.pop("password", None)
                    account.update(hash_password(reset_admin_password))
                    break
            save_users(users)
            st.success(f"Password reset for '{reset_admin}'.")
            st.rerun()

    st.markdown("**Delete master/admin account**")
    if len(master_users) <= 1:
        st.info("At least one master/admin account must remain.")
        return
    current_username = user["username"]
    deletable_admins = [account for account in master_users if account["username"] != current_username]
    if not deletable_admins:
        st.info("You cannot delete the currently logged-in account.")
        return
    with st.form("delete_admin_account"):
        delete_admin = st.selectbox(
            "Master/admin account to delete",
            options=[account["username"] for account in deletable_admins],
            key="admin_delete_account",
        )
        confirm_admin_delete = st.checkbox("I understand this will remove the account.")
        delete_admin_ok = st.form_submit_button("Delete master/admin account")
    if delete_admin_ok:
        if delete_admin == current_username:
            st.error("You cannot delete the currently logged-in account.")
        elif len(master_users) <= 1:
            st.error("At least one master/admin account must remain.")
        elif not confirm_admin_delete:
            st.error("Please confirm account deletion.")
        elif delete_user(delete_admin):
            st.success(f"Deleted master/admin account '{delete_admin}'.")
            st.rerun()
        else:
            st.error("Account could not be found.")


def render_staff_access():
    st.subheader("Staff Accounts")
    users = load_users()
    location_ids = [loc.id for loc in locations]
    locations_by_id = {loc.id: loc for loc in locations}

    def location_label(location_id: str) -> str:
        loc = locations_by_id.get(location_id)
        return f"{loc.name} ({location_id})" if loc else location_id

    staff_users = [account for account in users if account["role"] == "staff"]
    if staff_users:
        rows = []
        for account in staff_users:
            assigned_locations = account.get("authorized_locations", [])
            rows.append(
                {
                    "Username": account["username"],
                    "Authorized locations": ", ".join(location_label(location_id) for location_id in assigned_locations)
                    or "None",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No staff accounts have been created yet.")

    render_master_accounts(users)

    st.markdown("**Create staff account**")
    with st.form("create_staff_account"):
        new_username = st.text_input("New username")
        new_password = st.text_input("New password", type="password")
        new_locations = st.multiselect(
            "Authorized locations",
            options=location_ids,
            format_func=location_label,
            key="new_staff_locations",
        )
        create_ok = st.form_submit_button("Create staff account")
    if create_ok:
        new_username = new_username.strip()
        password_error = validate_password(new_password)
        if not new_username:
            st.error("Please enter a username.")
        elif password_error:
            st.error(password_error)
        elif not new_locations:
            st.error("Please assign at least one location.")
        elif any(account["username"].lower() == new_username.lower() for account in users):
            st.error("That username already exists.")
        else:
            users.append(
                {
                    "username": new_username,
                    "role": "staff",
                    "authorized_locations": new_locations,
                    **hash_password(new_password),
                }
            )
            save_users(users)
            st.success(f"Created staff account '{new_username}'.")
            st.rerun()

    if not staff_users:
        return

    st.markdown("**Update staff location access**")
    selected_staff = st.selectbox(
        "Staff account",
        options=[account["username"] for account in staff_users],
        key="staff_access_account",
    )
    selected_account = next(account for account in staff_users if account["username"] == selected_staff)
    current_locations = [
        location_id
        for location_id in selected_account.get("authorized_locations", [])
        if location_id in locations_by_id
    ]
    with st.form("update_staff_access"):
        updated_locations = st.multiselect(
            "Authorized locations",
            options=location_ids,
            default=current_locations,
            format_func=location_label,
            key="updated_staff_locations",
        )
        update_ok = st.form_submit_button("Save staff access")
    if update_ok:
        if not updated_locations:
            st.error("Please assign at least one location.")
        else:
            for account in users:
                if account["username"] == selected_staff:
                    account["authorized_locations"] = updated_locations
                    break
            save_users(users)
            st.success(f"Updated location access for '{selected_staff}'.")
            st.rerun()

    st.markdown("**Reset staff password**")
    reset_staff = st.selectbox(
        "Staff account",
        options=[account["username"] for account in staff_users],
        key="staff_password_reset_account",
    )
    with st.form("reset_staff_password"):
        reset_password = st.text_input("New password", type="password")
        staff_password_has_outer_space = reset_password != reset_password.strip()
        if staff_password_has_outer_space:
            st.warning("This password begins or ends with whitespace.")
        confirm_staff_whitespace = st.checkbox(
            "Save staff password with leading/trailing whitespace",
            disabled=not staff_password_has_outer_space,
        )
        reset_ok = st.form_submit_button("Reset password")
    if reset_ok:
        password_error = validate_password(reset_password)
        if password_error:
            st.error(password_error)
        elif staff_password_has_outer_space and not confirm_staff_whitespace:
            st.error("Please confirm before saving a password with leading or trailing whitespace.")
        else:
            for account in users:
                if account["username"] == reset_staff:
                    account.pop("password", None)
                    account.update(hash_password(reset_password))
                    break
            save_users(users)
            st.success(f"Password reset for '{reset_staff}'.")
            st.rerun()

    st.markdown("**Delete staff account**")
    with st.form("delete_staff_account"):
        delete_staff = st.selectbox(
            "Staff account to delete",
            options=[account["username"] for account in staff_users],
            key="staff_delete_account",
        )
        confirm_staff_delete = st.checkbox("I understand this will remove the staff account.")
        delete_staff_ok = st.form_submit_button("Delete staff account")
    if delete_staff_ok:
        if not confirm_staff_delete:
            st.error("Please confirm account deletion.")
        elif delete_user(delete_staff):
            st.success(f"Deleted staff account '{delete_staff}'.")
            st.rerun()
        else:
            st.error("Account could not be found.")


def render_admin_diagnostics():
    st.subheader("Admin Diagnostics")
    users = load_users()
    master_users = [account for account in users if require_role(account, {"master", "admin"})]
    staff_users = [account for account in users if account["role"] == "staff"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", len(users))
    c2.metric("Master accounts", len(master_users))
    c3.metric("Staff accounts", len(staff_users))
    c4.metric("Locations", len(locations))
    st.caption(f"User store: {user_store_mode()}")

    location_ids = {loc.id for loc in locations}
    rows = []
    for account in users:
        authorized_locations = account.get("authorized_locations", [])
        account_type = "hashed" if all(account.get(field) for field in ("password_hash", "salt", "iterations")) else "plaintext"
        validation = validate_user_record(account["username"])
        if account["role"] == "master" and authorized_locations == ["*"]:
            authorized_count = "all"
            location_status = "OK"
        else:
            invalid_locations = [location_id for location_id in authorized_locations if location_id not in location_ids]
            authorized_count = len(authorized_locations)
            location_status = "OK" if not invalid_locations else f"Unknown: {', '.join(invalid_locations)}"
        rows.append(
            {
                "Username": account["username"],
                "Role": account["role"],
                "Account type": account_type,
                "Authorized locations count": authorized_count,
                "Record check": "Pass" if validation["passed"] else "Fail",
                "Reason": validation["reason"],
                "Location IDs": location_status,
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_location_management():
    st.subheader("Locations")
    rows = [
        {
            "id": loc.id,
            "name": loc.name,
            "zip_code": loc.zip_code,
            "country_code": loc.country_code,
            "timezone": loc.timezone,
        }
        for loc in locations
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**Add location**")
    with st.form("add_location"):
        new_id = st.text_input("Location ID")
        new_name = st.text_input("Name")
        new_zip = st.text_input("Zip code")
        new_country = st.text_input("Country code", value="US")
        new_timezone = st.text_input("Timezone", value="America/New_York")
        add_ok = st.form_submit_button("Add location")
    if add_ok:
        new_location = {
            "id": new_id.strip(),
            "name": new_name.strip(),
            "zip_code": new_zip.strip(),
            "country_code": new_country.strip() or "US",
            "timezone": new_timezone.strip() or "America/New_York",
        }
        if not new_location["id"]:
            st.error("Please enter a location ID.")
        elif not new_location["name"]:
            st.error("Please enter a location name.")
        elif not new_location["zip_code"]:
            st.error("Please enter a zip code.")
        elif any(loc.id == new_location["id"] for loc in locations):
            st.error("That location ID already exists.")
        else:
            save_locations(rows + [new_location])
            st.success(f"Added location '{new_location['id']}'.")
            st.rerun()

    if not locations:
        return

    st.markdown("**Edit location**")
    selected_location_id = st.selectbox(
        "Location to edit",
        options=[loc.id for loc in locations],
        format_func=lambda loc_id: next((loc.name for loc in locations if loc.id == loc_id), loc_id),
        key="edit_location_id",
    )
    selected_location = next(loc for loc in locations if loc.id == selected_location_id)
    with st.form("edit_location"):
        edit_name = st.text_input("Name", value=selected_location.name)
        edit_zip = st.text_input("Zip code", value=selected_location.zip_code)
        edit_country = st.text_input("Country code", value=selected_location.country_code)
        edit_timezone = st.text_input("Timezone", value=selected_location.timezone)
        edit_ok = st.form_submit_button("Save location")
    if edit_ok:
        if not edit_name.strip():
            st.error("Please enter a location name.")
        elif not edit_zip.strip():
            st.error("Please enter a zip code.")
        else:
            updated_rows = []
            for row in rows:
                if row["id"] == selected_location_id:
                    row = {
                        "id": selected_location_id,
                        "name": edit_name.strip(),
                        "zip_code": edit_zip.strip(),
                        "country_code": edit_country.strip() or "US",
                        "timezone": edit_timezone.strip() or "America/New_York",
                    }
                updated_rows.append(row)
            save_locations(updated_rows)
            st.success(f"Updated location '{selected_location_id}'.")
            st.rerun()



t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    [
        "Prediction",
        "Metrics",
        "Model Monitoring",
        "Data Management",
        "Staff Access",
        "Location Management",
        "Admin Diagnostics",
    ]
)
with t1:
    render_prediction()
with t2:
    render_metrics()
with t3:
    render_model_monitoring()
with t4:
    render_data_ops()
with t5:
    render_staff_access()
with t6:
    render_location_management()
with t7:
    render_admin_diagnostics()
