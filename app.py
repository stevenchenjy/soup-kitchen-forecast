from datetime import date, datetime, timezone

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
from src.config import (
    DATE_COL,
    ESTIMATED_WASTE_REDUCTION_RATE,
    TARGET_COL,
    ForecastHorizonError,
    ForecastTargetDateError,
    ServiceDateParseError,
    model_file_for_location,
    parse_service_date,
)
from src.data_admin import (
    RetrainingQueueError,
    attendance_store_mode,
    delete_record,
    load_clean_data,
    save_clean_data,
    upsert_record,
)
from src.f6_charts import (
    build_absolute_error_chart,
    build_actual_vs_predicted_chart,
)
from src.f6_monitoring import (
    BacktestChartError,
    BacktestSummaryError,
    F6IntegrityError,
    active_f6_package,
    build_operational_impact,
    build_f6_monitoring_report,
    f6_training_status,
    format_dashboard_date,
    load_verified_backtest_chart_series,
    load_verified_backtest_summary,
    retraining_status_label,
)
from src.location_config import save_locations, list_locations
from src.model_training_runs import (
    get_retrain_state,
    latest_successful_training_run,
    latest_training_run,
    model_training_run_store_mode,
    model_training_run_store_fingerprint,
)
from src.prediction_logs import (
    load_prediction_logs,
    prediction_log_store_mode,
    save_prediction_log,
    update_prediction_logs_with_actual,
)
from src.predictor import VisitorPredictor, WeatherForecastUnavailableError

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
        "User accounts are using local data/users.json storage. On Streamlit Cloud this is not shared between "
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
predictor = None
model_load_error = None
if model_path.exists():
    try:
        predictor = VisitorPredictor(str(model_path))
    except Exception as exc:
        model_load_error = exc

active_package = None
f6_integrity_error = None
if predictor is not None:
    try:
        active_package = active_f6_package(predictor)
    except F6IntegrityError as exc:
        f6_integrity_error = str(exc)
else:
    f6_integrity_error = "The active F6 model could not be loaded."



def render_prediction():
    st.subheader(f"Prediction - {selected_name}")
    if active_package is None:
        st.error("F6 integrity error")
        st.caption(f6_integrity_error or "The active F6 contract is unavailable.")
        return

    st.caption(
        "Recommended meals include a built-in safety margin based on expected attendance."
    )
    custom_date = st.text_input(
        "Target service date (Sat/Sun, YYYY-MM-DD)", value=""
    )

    if st.button("Generate prediction", type="primary"):
        try:
            normalized_date = parse_service_date(custom_date) if custom_date else None
            if normalized_date is not None and normalized_date.isoformat() != custom_date.strip():
                st.info(f"Using service date: {normalized_date.isoformat()}")
            pred = predictor.predict_next(
                target_date=normalized_date, meal_buffer_pct=None
            )
            st.success(f"Recommendation ready for {pred.service_date:%Y-%m-%d}.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Service date", pred.service_date.strftime("%Y-%m-%d"))
            c2.metric("Expected visitors", f"{pred.predicted_visitors:.1f}")
            c3.metric("Recommended meals", f"{pred.suggested_meals}")
            try:
                save_prediction_log(location_id, pred, created_by=user["username"], source_app="admin")
            except Exception:
                st.warning("Prediction was generated, but monitoring log could not be saved.")
        except ServiceDateParseError:
            st.error("Please enter the service date in YYYY-MM-DD format, for example 2026-07-04.")
        except ForecastHorizonError:
            st.error("Forecasts are only available within 16 days because weather forecasts are not reliable beyond that range.")
        except ForecastTargetDateError as exc:
            st.error(str(exc))
        except WeatherForecastUnavailableError:
            st.error("Weather forecast data is unavailable for this date. Please try again later or choose another date within the forecast window.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


@st.fragment(run_every=60)
def render_training_status():
    st.markdown("### Training Status")
    try:
        retrain_state = get_retrain_state(location_id)
        latest_run = latest_training_run(location_id)
        latest_success = latest_successful_training_run(location_id)
        training = f6_training_status(
            active_package,
            retrain_state=retrain_state,
            latest_run=latest_run,
            latest_successful_run=latest_success,
        )
    except Exception as exc:
        st.warning(f"Training status could not be loaded: {exc}")
        return

    t1, t2, t3 = st.columns(3)
    t1.caption("Needs Retraining")
    t1.markdown(f"**{'Yes' if training['needs_retraining'] else 'No'}**")
    t2.caption("Last Attendance Update")
    t2.markdown(
        f"**{format_dashboard_date((retrain_state or {}).get('last_attendance_updated_at'))}**"
    )
    t3.caption("Retraining Status")
    t3.markdown(f"**{retraining_status_label(training['status'])}**")
    if training["deployment_mismatch"]:
        st.warning(training["message"])
    else:
        st.info(training["message"])
    checked_at = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    st.caption(
        "Status source "
        f"`{model_training_run_store_fingerprint()}` · checked {checked_at} · "
        "refreshes every minute"
    )


def render_model_monitoring():
    st.subheader("Model Monitoring")
    st.caption(f"Prediction log store: {prediction_log_store_mode()}")
    st.caption(f"Training run store: {model_training_run_store_mode()}")

    if active_package is None:
        st.error("F6 integrity error")
        st.caption(f6_integrity_error or "The active F6 contract is unavailable.")
        return

    try:
        backtest = load_verified_backtest_summary(active_package)
    except BacktestSummaryError as exc:
        backtest = None
        backtest_error = exc
    else:
        backtest_error = None

    feature_hash = active_package.feature_order_sha256
    if active_package.training_cutoff is None:
        trained_through = "an unavailable date"
    else:
        cutoff = date.fromisoformat(active_package.training_cutoff)
        trained_through = f"{cutoff.strftime('%B')} {cutoff.day}, {cutoff.year}"
    st.markdown("### Active Model")
    st.success(
        f"Forecast model active · Trained through {trained_through} · "
        "Raw Q80 recommendation"
    )
    with st.expander("Technical details", expanded=False):
        st.markdown(f"**Package ID:** `{active_package.package_id}`")
        st.markdown(f"**Schema Version:** {active_package.schema_version}")
        st.markdown(f"**Feature Set:** `{active_package.feature_set_id}`")
        st.markdown(
            f"**Feature hash:** `{feature_hash[:12]}…{feature_hash[-8:]}`"
        )
        st.markdown(
            f"**Recommendation Policy:** `{active_package.recommendation_policy_id}`"
        )

    def number(value):
        return "—" if value is None else f"{value:.2f}"

    def rate(value):
        return "—" if value is None else f"{value * 100:.1f}%"

    st.markdown("### Model Performance")
    if backtest_error is not None:
        st.error(f"Verified historical backtest unavailable: {backtest_error}")
    if backtest is not None:
        metrics = backtest["metrics"]
        cutoff = date.fromisoformat(backtest["attendance_cutoff"])
        if backtest["is_reference_backtest"]:
            st.caption(
                "Verified reference backtest for the locked F6 method using "
                "attendance through "
                f"{cutoff.strftime('%B')} {cutoff.day}, {cutoff.year}."
            )
        else:
            st.caption(
                "Origin-aware historical backtest using attendance through "
                f"{cutoff.strftime('%B')} {cutoff.day}, {cutoff.year}."
            )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", number(metrics["mae"]))
        m2.metric("Median Absolute Error", number(metrics["median_absolute_error"]))
        m3.metric("RMSE", number(metrics["rmse"]))
        m4.metric("Mean Signed Error", number(metrics["mean_signed_error"]))
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Underprediction Rate", rate(metrics["underprediction_rate"]))
        m6.metric("Q80 Empirical Coverage", rate(metrics["q80_empirical_coverage"]))
        m7.metric("Mean Over-Preparation", number(metrics["mean_over_preparation"]))
        m8.metric("Mean Under-Preparation", number(metrics["mean_under_preparation"]))
        s1, s2 = st.columns(2)
        s1.metric("Saturday MAE", number(backtest["segments"]["Saturday"]["mae"]))
        s2.metric("Sunday MAE", number(backtest["segments"]["Sunday"]["mae"]))
        h1, h2, h5 = st.columns(3)
        h1.metric("H1 MAE", number(backtest["horizons"]["H1"]["mae"]))
        h2.metric("H2 MAE", number(backtest["horizons"]["H2"]["mae"]))
        h5.metric("H5 MAE", number(backtest["horizons"]["H5"]["mae"]))

        try:
            chart_series = load_verified_backtest_chart_series(active_package)
        except BacktestChartError:
            st.warning("Historical performance charts are temporarily unavailable.")
        else:
            st.markdown("#### Actual vs Predicted")
            st.altair_chart(
                build_actual_vs_predicted_chart(chart_series),
                use_container_width=True,
            )
            st.caption(
                "Origin-aware historical predictions through July 12, 2026."
            )
            st.markdown("#### Absolute Error Over Time")
            st.altair_chart(
                build_absolute_error_chart(chart_series),
                use_container_width=True,
            )

    st.markdown("### Live Performance")
    try:
        logs = load_prediction_logs(location_id=location_id, limit=5000)
    except Exception as exc:
        logs = None
        report = None
        st.error(f"Production monitoring data could not be loaded: {exc}")
    else:
        report = build_f6_monitoring_report(
            predictor,
            logs,
            load_error=model_load_error,
        )
        if not report["integrity_ok"]:
            st.error("F6 integrity error")
            for message in report["integrity_errors"]:
                st.caption(message)
            report = None

    if report is not None:
        if report["prediction_count"] == 0:
            st.info("No live production predictions are available yet.")
        else:
            stage_labels = {
                "INSUFFICIENT_DATA": "Insufficient data",
                "EARLY_SIGNAL": "Early signal",
                "INITIAL_REVIEW": "Initial review",
                "STABLE_REVIEW": "Stable review",
            }
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Production Predictions", report["prediction_count"])
            c2.metric("Reconciled Predictions", report["reconciled_count"])
            c3.metric("Unreconciled Predictions", report["unreconciled_count"])
            c4.metric("Monitoring Stage", stage_labels[report["stage"]])

            live_metrics = report["metrics"]
            if report["reconciled_count"] <= 3:
                st.info(
                    "Insufficient production data. Live metrics will appear after actual attendance is recorded."
                )
            else:
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("Live MAE", number(live_metrics["mae"]))
                l2.metric("Live RMSE", number(live_metrics["rmse"]))
                l3.metric("Live Mean Signed Error", number(live_metrics["mean_signed_error"]))
                l4.metric("Live Underprediction Rate", rate(live_metrics["underprediction_rate"]))
                l5, l6, l7 = st.columns(3)
                l5.metric("Live Q80 Coverage", rate(live_metrics["q80_empirical_coverage"]))
                l6.metric("Live Mean Over-Preparation", number(live_metrics["mean_over_preparation"]))
                l7.metric("Live Mean Under-Preparation", number(live_metrics["mean_under_preparation"]))

            if report["integrity_alerts"]:
                st.markdown("**Integrity Alerts**")
                for message in report["integrity_alerts"]:
                    st.warning(message)
            if report["reconciled_count"] >= 4 and report["performance_alerts"]:
                st.markdown("**Performance Alerts**")
                for message in report["performance_alerts"]:
                    st.warning(message)

            st.markdown("**Latest Production Outcomes**")
            outcome_rows = []
            for row in report["latest_outcomes"]:
                predicted = row.get("predicted_visitors")
                actual = row.get("actual_visitors")
                absolute_error = None
                if predicted is not None and actual is not None:
                    absolute_error = abs(float(predicted) - float(actual))
                outcome_rows.append(
                    {
                        "Service date": row.get("service_date"),
                        "Segment": str(row.get("model_segment") or "").upper(),
                        "Horizon": row.get("service_horizon"),
                        "Expected visitors": (
                            None if predicted is None else round(float(predicted), 2)
                        ),
                        "Raw Q80": (
                            None
                            if row.get("predicted_quantile") is None
                            else round(float(row["predicted_quantile"]), 2)
                        ),
                        "Recommended meals": row.get("suggested_meals"),
                        "Actual visitors": actual,
                        "Absolute error": (
                            None if absolute_error is None else round(absolute_error, 2)
                        ),
                    }
                )
            st.dataframe(outcome_rows, use_container_width=True, hide_index=True)

    st.markdown("### Operational Impact")
    try:
        attendance_rows = len(load_clean_data(location_id))
    except Exception as exc:
        attendance_rows = None
        st.warning(f"Attendance total could not be loaded: {exc}")
    if logs is None:
        impact = None
    else:
        impact = build_operational_impact(attendance_rows, logs)
    o1, o2, o3 = st.columns(3)
    o1.metric(
        "Attendance Rows",
        "—" if attendance_rows is None else attendance_rows,
    )
    o2.metric(
        "Total Prediction Logs",
        "—" if impact is None else impact["prediction_log_count"],
    )
    o3.metric(
        "Logs Reconciled with Actuals",
        "—" if impact is None else impact["reconciled_log_count"],
    )
    o4, o5 = st.columns(2)
    o4.metric(
        "Estimated Food Saved",
        "—" if impact is None else f"{impact['estimated_food_saved_meals']:.1f} meals",
    )
    o5.metric(
        "Estimated CO₂e Reduction",
        "—" if impact is None else f"{impact['estimated_co2e_reduction_kg']:.1f} kg",
    )
    st.caption(
        f"Cumulative operational estimates across project records using a "
        f"{ESTIMATED_WASTE_REDUCTION_RATE:.0%} meal-waste reduction assumption."
    )

    render_training_status()



def render_data_ops():
    st.subheader(f"Data Management - {selected_name}")
    df = load_clean_data(location_id)

    st.markdown("### Attendance Records")
    st.caption("View and filter historical attendance data.")
    q1, q2 = st.columns(2)
    with q1:
        start = st.date_input("Start date", value=df[DATE_COL].min().date() if not df.empty else None)
    with q2:
        end = st.date_input("End date", value=df[DATE_COL].max().date() if not df.empty else None)

    if start and end and not df.empty:
        show = df[(df[DATE_COL].dt.date >= start) & (df[DATE_COL].dt.date <= end)].copy()
    else:
        show = df.copy()

    st.markdown("**Recent Attendance Records**")
    attendance_display = show[[DATE_COL, TARGET_COL]].copy()
    if not attendance_display.empty:
        attendance_display[DATE_COL] = attendance_display[DATE_COL].dt.strftime("%Y-%m-%d")
    attendance_display = attendance_display.rename(
        columns={DATE_COL: "Service date", TARGET_COL: "Actual visitors served"}
    )
    st.dataframe(attendance_display, use_container_width=True, height=260, hide_index=True)

    st.markdown("### Add or Correct Attendance")
    st.caption("Add a new attendance record or correct an existing one.")
    a1, a2, a3 = st.columns([2, 2, 1])
    with a1:
        add_date = st.date_input("Service date", value=None, key="add_date")
    with a2:
        add_visitors = st.number_input("Actual visitors served", min_value=0, max_value=10000, value=120, step=1)
    with a3:
        if st.button("Save Attendance"):
            if add_date is None:
                st.error("Please select date")
            else:
                saved = False
                try:
                    upsert_record(str(add_date), int(add_visitors), location_id)
                    saved = True
                except RetrainingQueueError as exc:
                    saved = True
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Attendance could not be saved: {exc}")
                if saved:
                    monitoring_updated = True
                    try:
                        update_prediction_logs_with_actual(
                            location_id, str(add_date), int(add_visitors)
                        )
                    except Exception:
                        monitoring_updated = False
                        st.warning(
                            "Attendance was saved, but monitoring log could not be updated."
                        )
                    st.success("Saved")
                    if monitoring_updated:
                        st.rerun()

    with st.expander("Advanced Administration", expanded=False):
        st.markdown("#### Delete Attendance Record")
        del_options = [d.strftime("%Y-%m-%d") for d in df[DATE_COL].sort_values()] if not df.empty else []
        del_date = st.selectbox("Service date to delete", options=del_options, index=None, placeholder="Select date")
        if st.button("Delete Attendance Record"):
            if del_date:
                deleted = False
                try:
                    delete_record(del_date, location_id)
                    deleted = True
                except RetrainingQueueError as exc:
                    deleted = True
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Attendance could not be deleted: {exc}")
                if deleted:
                    st.success(f"Deleted {del_date}")
                    st.rerun()
            else:
                st.error("Please select date")

        st.markdown("#### Bulk Attendance Editor")
        st.caption(
            "Edit multiple attendance records at once. "
            "Changes are not saved until you click Save Bulk Edits."
        )
        edit_df = df[[DATE_COL, TARGET_COL]].copy()
        if not edit_df.empty:
            edit_df[DATE_COL] = edit_df[DATE_COL].dt.strftime("%Y-%m-%d")
        edit_df = st.data_editor(
            edit_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                DATE_COL: st.column_config.TextColumn("Service date"),
                TARGET_COL: st.column_config.NumberColumn("Actual visitors served", min_value=0, step=1),
            },
        )
        if st.button("Save Bulk Edits"):
            saved = False
            try:
                save_clean_data(edit_df, location_id)
                saved = True
            except RetrainingQueueError as exc:
                saved = True
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Attendance could not be saved: {exc}")
            if saved:
                st.success("Saved")
                st.rerun()

        st.markdown("#### Production Retraining")
        st.caption(
            "Retraining uses the guarded publication workflow. Current status is "
            "shown in Model Monitoring."
        )


def render_master_accounts(users, location_display_row):
    st.subheader("Master/Admin Accounts")
    master_users = [account for account in users if require_role(account, {"master", "admin"})]
    if master_users:
        st.dataframe([location_display_row(account) for account in master_users], use_container_width=True, hide_index=True)
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

    def assigned_locations_label(account: dict) -> str:
        role = account.get("role")
        assigned_locations = account.get("authorized_locations", [])
        if role == "master":
            return "All Locations"
        if not assigned_locations:
            return "None"
        labels = []
        for location_id in assigned_locations:
            loc = locations_by_id.get(location_id)
            labels.append(loc.name if loc else f"Unknown location ({location_id})")
        return ", ".join(labels)

    def location_display_row(account: dict) -> dict:
        validation = validate_user_record(account["username"])
        return {
            "Username": account["username"],
            "Role": account.get("role", ""),
            "Assigned Locations": assigned_locations_label(account),
            "Status": "Active" if validation["passed"] else "Invalid",
        }

    staff_users = [account for account in users if account["role"] == "staff"]
    if staff_users:
        st.dataframe([location_display_row(account) for account in staff_users], use_container_width=True, hide_index=True)
    else:
        st.info("No staff accounts have been created yet.")

    render_master_accounts(users, location_display_row)

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
    st.caption(f"Attendance store: {attendance_store_mode()}")

    rows = []
    for account in users:
        account_type = "hashed" if all(account.get(field) for field in ("password_hash", "salt", "iterations")) else "plaintext"
        validation = validate_user_record(account["username"])
        rows.append(
            {
                "Username": account["username"],
                "Account type": account_type,
                "Record check": "Pass" if validation["passed"] else "Fail",
                "Reason": validation["reason"],
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



t1, t2, t3, t4, t5, t6 = st.tabs(
    [
        "Prediction",
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
    render_model_monitoring()
with t3:
    render_data_ops()
with t4:
    render_staff_access()
with t5:
    render_location_management()
with t6:
    render_admin_diagnostics()
