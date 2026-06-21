import streamlit as st

from src.auth import authenticate_user, get_authorized_locations, get_user, require_role
from src.config import (
    ESTIMATED_WASTE_REDUCTION_RATE,
    ForecastTargetDateError,
    KG_CO2E_PER_KG_FOOD_WASTE,
    MEAL_WEIGHT_KG,
    TARGET_COL,
    model_file_for_location,
)
from src.data_admin import (
    delete_latest_staff_created_attendance,
    latest_staff_created_attendance,
    load_clean_data,
    upsert_record,
)
from src.location_config import list_locations
from src.prediction_logs import save_prediction_log, update_prediction_logs_with_actual
from src.predictor import VisitorPredictor, WeatherForecastUnavailableError

st.set_page_config(page_title="Staff Meal Prep Assistant", layout="centered")



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
st.caption("Per-location view with independent data/model storage")

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
df = load_clean_data(location_id)
predictor = None
model_load_error = None
if model_path.exists():
    try:
        predictor = VisitorPredictor(str(model_path))
    except Exception as exc:
        model_load_error = exc
delete_message = st.session_state.pop("staff_delete_message", None)
if delete_message:
    st.success(delete_message)

st.subheader(f"Daily Actions - {selected_name}")
if predictor is None:
    if model_path.exists() and model_load_error is not None:
        st.warning("Model file is incompatible. Please retrain model.")
    else:
        st.warning("Forecast is not ready for this location. Please contact an admin.")
else:
    buf = st.slider(
        "Extra meals safety buffer (%)",
        0,
        30,
        8,
        1,
        help="This adds a small safety margin so the kitchen prepares a few extra meals in case more visitors arrive than predicted.",
    )
    custom_date = st.text_input("Target service date (Saturday/Sunday, YYYY-MM-DD)", value="")
    if st.button("Get meal recommendation", type="primary"):
        try:
            pred = predictor.predict_next(target_date=custom_date or None, meal_buffer_pct=buf / 100.0)
        except ForecastTargetDateError:
            st.error("Forecasts are only available within 16 days because weather forecasts are not reliable beyond that range.")
        except WeatherForecastUnavailableError:
            st.error("Weather forecast data is unavailable for this date. Please choose a date within the forecast window.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.success(f"Recommendation ready for {pred.service_date:%Y-%m-%d}.")
            estimated_food_saved = pred.suggested_meals * ESTIMATED_WASTE_REDUCTION_RATE
            estimated_carbon_reduced = estimated_food_saved * MEAL_WEIGHT_KG * KG_CO2E_PER_KG_FOOD_WASTE
            c1, c2 = st.columns(2)
            c1.metric("Recommended Meals", f"{pred.suggested_meals}")
            c2.metric("Expected Visitors", f"{pred.predicted_visitors:.1f}")
            c3, c4 = st.columns(2)
            c3.metric("Estimated Food Saved", f"{estimated_food_saved:.1f} meals of food")
            c4.metric("Estimated Carbon Reduced", f"{estimated_carbon_reduced:.1f} kg CO2e")
            st.caption(
                f"These are planning estimates based on a {ESTIMATED_WASTE_REDUCTION_RATE:.0%} "
                "waste-reduction assumption."
            )
            try:
                save_prediction_log(location_id, pred, created_by=user["username"], source_app="staff")
            except Exception:
                st.warning("Prediction was generated, but monitoring log could not be saved.")

st.subheader("After Service")
add_date = st.date_input("Service date", value=None, key="staff_add_date")
add_visitors = st.number_input("Actual visitors served", min_value=0, max_value=10000, value=120, step=1)
if st.button("Add / Update"):
    if add_date is not None:
        upsert_record(str(add_date), int(add_visitors), location_id, changed_by=user["username"])
        monitoring_updated = True
        try:
            update_prediction_logs_with_actual(location_id, str(add_date), int(add_visitors))
        except Exception:
            monitoring_updated = False
            st.warning("Attendance was saved, but monitoring log could not be updated.")
        st.success("Saved.")
        if monitoring_updated:
            st.rerun()

if not df.empty:
    st.markdown("**Recent visitor counts**")
    recent_df = df[["service_date", TARGET_COL]].sort_values("service_date", ascending=False).head(5).copy()
    recent_df["service_date"] = recent_df["service_date"].dt.strftime("%Y-%m-%d")
    recent_df = recent_df.rename(columns={"service_date": "Service date", TARGET_COL: "Actual visitors served"})
    st.dataframe(recent_df, use_container_width=True, hide_index=True, height=300)

    with st.expander("View full attendance history", expanded=False):
        filter_start, filter_end = st.columns(2)
        with filter_start:
            history_start = st.date_input(
                "Start date",
                value=df["service_date"].min().date(),
                key="staff_history_start",
            )
        with filter_end:
            history_end = st.date_input(
                "End date",
                value=df["service_date"].max().date(),
                key="staff_history_end",
            )

        history_df = df[["service_date", TARGET_COL]].copy()
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


st.markdown("**Delete My Latest Attendance Entry**")
latest_entry = latest_staff_created_attendance(location_id, user["username"])
if latest_entry is None:
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
            deleted = delete_latest_staff_created_attendance(location_id, user["username"])
            if deleted is None:
                st.error("That attendance entry is no longer available to delete.")
            else:
                st.session_state["staff_delete_message"] = (
                    f"Deleted your attendance entry for {deleted['service_date']}."
                )
                st.rerun()
