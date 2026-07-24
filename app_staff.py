from urllib.error import HTTPError, URLError

import streamlit as st

from src.auth import authenticate_user, get_authorized_locations, get_user, require_role
from src.config import (
    ForecastHorizonError,
    ForecastTargetDateError,
    ServiceDateParseError,
    TARGET_COL,
    model_file_for_location,
    parse_service_date,
)
from src.data_admin import (
    RetrainingQueueError,
    delete_latest_staff_created_attendance,
    latest_staff_created_attendance,
    load_clean_data,
    load_recent_attendance,
    upsert_record,
)
from src.f6_monitoring import F6IntegrityError, active_f6_package
from src.location_config import list_locations
from src.prediction_logs import save_prediction_log, update_prediction_logs_with_actual
from src.predictor import VisitorPredictor, WeatherForecastUnavailableError

st.set_page_config(page_title="Staff Meal Prep Assistant", layout="centered")

ATTENDANCE_UNAVAILABLE_MESSAGE = "Attendance data is temporarily unavailable. Please refresh in a moment."
ATTENDANCE_SAVE_ERROR_MESSAGE = "Could not save attendance right now. Please try again."



def login_gate() -> None:
    st.title("Login")
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

if not require_role(user, {"master", "staff"}):
    st.error("No permission.")
    st.stop()

all_locations = list_locations()
if not all_locations:
    st.title("Staff Meal Prep Assistant")
    st.error("No locations are configured yet. Please contact a master user.")
    st.stop()

locations = get_authorized_locations(user, all_locations)
loc_names = {loc.name: loc.id for loc in locations}

st.title("Staff Meal Prep Assistant")
st.caption("Plan meals and record attendance for your location.")

sidebar = st.sidebar
sidebar.markdown(f"**User:** {user['username']} ({user['role']})")
if not locations:
    st.error("No authorized locations are assigned to your account. Please contact a master user.")
    if sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.stop()

selected_name = sidebar.selectbox("Location", options=list(loc_names.keys()))
location_id = loc_names[selected_name]
if location_id not in {loc.id for loc in locations}:
    st.error("You are not authorized to access this location.")
    st.stop()

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
if predictor is not None:
    try:
        active_package = active_f6_package(predictor)
    except F6IntegrityError:
        pass
delete_message = st.session_state.pop("staff_delete_message", None)
if delete_message:
    st.success(delete_message)

st.subheader(f"Daily Actions - {selected_name}")
if active_package is None:
    st.error("Forecast unavailable")
    st.caption(
        "The forecasting system failed an integrity check. Please contact an administrator."
    )
else:
    st.caption(
        "Recommended meals include a built-in safety margin based on expected attendance."
    )
    custom_date = st.text_input("Target service date (Saturday/Sunday, YYYY-MM-DD)", value="")
    if st.button("Get meal recommendation", type="primary"):
        try:
            normalized_date = parse_service_date(custom_date) if custom_date else None
            if normalized_date is not None and normalized_date.isoformat() != custom_date.strip():
                st.info(f"Using service date: {normalized_date.isoformat()}")
            pred = predictor.predict_next(
                target_date=normalized_date, meal_buffer_pct=None
            )
        except ServiceDateParseError:
            st.error("Please enter the service date in YYYY-MM-DD format, for example 2026-07-04.")
        except ForecastHorizonError:
            st.error("Forecasts are only available within 16 days because weather forecasts are not reliable beyond that range.")
        except ForecastTargetDateError as exc:
            st.error(str(exc))
        except WeatherForecastUnavailableError:
            st.error("Weather forecast data is unavailable for this date. Please try again later or choose another date within the forecast window.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.success(f"Recommendation ready for {pred.service_date:%Y-%m-%d}.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Service date", pred.service_date.strftime("%Y-%m-%d"))
            c2.metric("Expected visitors", f"{pred.predicted_visitors:.1f}")
            c3.metric("Recommended meals", f"{pred.suggested_meals}")
            try:
                save_prediction_log(location_id, pred, created_by=user["username"], source_app="staff")
            except Exception:
                st.warning("Prediction was generated, but monitoring log could not be saved.")

st.subheader("After Service")
add_date = st.date_input("Service date", value=None, key="staff_add_date")
add_visitors = st.number_input("Actual visitors served", min_value=0, max_value=10000, value=120, step=1)
if st.button("Add / Update"):
    if add_date is not None:
        saved = False
        try:
            upsert_record(
                str(add_date),
                int(add_visitors),
                location_id,
                changed_by=user["username"],
                load_result=False,
            )
            saved = True
        except RetrainingQueueError as exc:
            saved = True
            st.warning(str(exc))
        except (TimeoutError, URLError, HTTPError):
            st.error(ATTENDANCE_SAVE_ERROR_MESSAGE)
        except Exception:
            st.error(ATTENDANCE_SAVE_ERROR_MESSAGE)
        if saved:
            try:
                update_prediction_logs_with_actual(location_id, str(add_date), int(add_visitors))
            except Exception:
                st.warning("Attendance was saved, but monitoring log could not be updated.")
            st.session_state.pop(f"staff_full_history_{location_id}", None)
            st.success("Saved.")

try:
    recent_attendance = load_recent_attendance(location_id, limit=7)
except (TimeoutError, URLError, HTTPError):
    recent_attendance = None
    st.warning(ATTENDANCE_UNAVAILABLE_MESSAGE)
except Exception:
    recent_attendance = None
    st.warning(ATTENDANCE_UNAVAILABLE_MESSAGE)

if recent_attendance is not None and not recent_attendance.empty:
    st.markdown("**Recent visitor counts**")
    recent_df = recent_attendance[["service_date", TARGET_COL]].copy()
    recent_df = recent_df.sort_values("service_date", ascending=False).head(7)
    recent_df["service_date"] = recent_df["service_date"].dt.strftime("%Y-%m-%d")
    recent_df = recent_df.rename(columns={"service_date": "Service date", TARGET_COL: "Actual visitors served"})
    recent_table_height = min(38 + (35 * len(recent_df)), 283)
    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
        height=recent_table_height,
    )

with st.expander("View full attendance history", expanded=False):
    history_key = f"staff_full_history_{location_id}"
    if st.button("Load full attendance history", key=f"load_{history_key}"):
        try:
            st.session_state[history_key] = load_clean_data(location_id)
        except (TimeoutError, URLError, HTTPError):
            st.session_state.pop(history_key, None)
            st.warning(ATTENDANCE_UNAVAILABLE_MESSAGE)
        except Exception:
            st.session_state.pop(history_key, None)
            st.warning(ATTENDANCE_UNAVAILABLE_MESSAGE)

    full_attendance = st.session_state.get(history_key)
    if full_attendance is not None and not full_attendance.empty:
        filter_start, filter_end = st.columns(2)
        with filter_start:
            history_start = st.date_input(
                "Start date",
                value=full_attendance["service_date"].min().date(),
                key="staff_history_start",
            )
        with filter_end:
            history_end = st.date_input(
                "End date",
                value=full_attendance["service_date"].max().date(),
                key="staff_history_end",
            )

        history_df = full_attendance[["service_date", TARGET_COL]].copy()
        if history_start > history_end:
            st.warning("Start date must be on or before end date.")
            history_df = history_df.iloc[0:0]
        else:
            history_df = history_df[
                (history_df["service_date"].dt.date >= history_start)
                & (history_df["service_date"].dt.date <= history_end)
            ]

        history_df = history_df.sort_values("service_date", ascending=False)
        history_df["service_date"] = history_df["service_date"].dt.strftime("%Y-%m-%d")
        history_df = history_df.rename(
            columns={"service_date": "Service date", TARGET_COL: "Actual visitors served"}
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    elif full_attendance is not None:
        st.caption("No attendance history is available for this location.")


st.markdown("**Delete My Latest Attendance Entry**")
try:
    latest_entry = latest_staff_created_attendance(location_id, user["username"])
except (TimeoutError, URLError, HTTPError):
    latest_entry = None
    latest_entry_unavailable = True
except Exception:
    latest_entry = None
    latest_entry_unavailable = True
else:
    latest_entry_unavailable = False

if latest_entry_unavailable:
    st.warning(ATTENDANCE_UNAVAILABLE_MESSAGE)
elif latest_entry is None:
    st.caption("No recent attendance entry found for your account.")
else:
    st.write("Latest attendance entry you created:")
    st.write(f"Date: {latest_entry['service_date']}")
    st.write(f"Visitors: {latest_entry['visitors']}")
    st.caption(
        "This removes the most recent attendance entry you created for this location. "
        "You can enter the correct value again afterward."
    )
    with st.form("delete_latest_staff_attendance"):
        confirm_delete = st.checkbox("Yes, delete this attendance entry")
        delete_ok = st.form_submit_button("Delete My Latest Attendance Entry")
    if delete_ok:
        if not confirm_delete:
            st.error("Please confirm before deleting this attendance entry.")
        else:
            try:
                deleted = delete_latest_staff_created_attendance(location_id, user["username"])
            except RetrainingQueueError as exc:
                st.warning(str(exc))
                st.session_state.pop(f"staff_full_history_{location_id}", None)
                st.session_state["staff_delete_message"] = (
                    f"Deleted your attendance entry for {latest_entry['service_date']}."
                )
                st.rerun()
            except (TimeoutError, URLError, HTTPError):
                st.error(ATTENDANCE_UNAVAILABLE_MESSAGE)
            except Exception:
                st.error(ATTENDANCE_UNAVAILABLE_MESSAGE)
            else:
                if deleted is None:
                    st.error("That attendance entry is no longer available to delete.")
                else:
                    st.session_state.pop(f"staff_full_history_{location_id}", None)
                    st.session_state["staff_delete_message"] = (
                        f"Deleted your attendance entry for {deleted['service_date']}."
                    )
                    st.rerun()
