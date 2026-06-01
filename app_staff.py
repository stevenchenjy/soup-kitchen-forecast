import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.auth import authenticate_user, get_authorized_locations, get_user, require_role
from src.config import (
    ESTIMATED_WASTE_REDUCTION_RATE,
    KG_CO2E_PER_KG_FOOD_WASTE,
    MEAL_WEIGHT_KG,
    TARGET_COL,
    model_file_for_location,
)
from src.data_admin import delete_record, load_clean_data, upsert_record
from src.location_config import list_locations
from src.prediction_logs import save_prediction_log, update_prediction_logs_with_actual
from src.predictor import VisitorPredictor

ROOT = Path(__file__).resolve().parent

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
predictor = VisitorPredictor(str(model_path)) if model_path.exists() else None

st.subheader(f"Daily Actions - {selected_name}")
if predictor is None:
    st.warning("Model not trained yet for this location.")
    if st.button("Train this location", type="primary"):
        with st.spinner("Training..."):
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_backtest.py"), "--location", location_id], cwd=ROOT)
        if r.returncode == 0:
            st.success("Training completed")
            st.rerun()
        else:
            st.error("Training failed")
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
        pred = predictor.predict_next(target_date=custom_date or None, meal_buffer_pct=buf / 100.0)
        st.success(
            f"Location: {location_id} | {pred.service_date:%Y-%m-%d} | Suggested meal prep: {pred.suggested_meals} "
            f"(Point: {pred.predicted_visitors:.1f}, Quantile: {pred.predicted_quantile:.1f}, Residual: +{pred.residual_buffer:.1f})"
        )
        estimated_food_saved = pred.suggested_meals * ESTIMATED_WASTE_REDUCTION_RATE
        estimated_carbon_reduced = estimated_food_saved * MEAL_WEIGHT_KG * KG_CO2E_PER_KG_FOOD_WASTE
        c1, c2 = st.columns(2)
        c1.metric("Estimated Food Saved", f"{estimated_food_saved:.1f} meals of food")
        c2.metric("Estimated Carbon Reduced", f"{estimated_carbon_reduced:.1f} kg CO2e")
        st.caption(
            f"These are planning estimates based on a {ESTIMATED_WASTE_REDUCTION_RATE:.0%} "
            "waste-reduction assumption."
        )
        try:
            save_prediction_log(location_id, pred, created_by=user["username"], source_app="staff")
        except Exception:
            st.warning("Prediction was generated, but monitoring log could not be saved.")

st.subheader("Quick Data Maintenance")
add_date = st.date_input("Service date", value=None, key="staff_add_date")
add_visitors = st.number_input("Visitors", min_value=0, max_value=10000, value=120, step=1)
if st.button("Add / Update"):
    if add_date is not None:
        upsert_record(str(add_date), int(add_visitors), location_id)
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
    del_options = [d.strftime("%Y-%m-%d") for d in df["service_date"].sort_values()]
    del_date = st.selectbox("Delete date", options=del_options)
    if st.button("Delete"):
        delete_record(del_date, location_id)
        st.success("Deleted.")
        st.rerun()

st.dataframe(df[["service_date", TARGET_COL]], use_container_width=True, height=300)

if st.button("Incremental retraining", type="primary"):
    with st.spinner("Training..."):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "retrain_incremental.py"), "--location", location_id], cwd=ROOT)
    if r.returncode == 0:
        st.success("Training completed.")
    else:
        st.error("Training failed.")
