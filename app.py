import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.auth import authenticate
from src.config import DATE_COL, TARGET_COL, artifact_dir_for_location, model_file_for_location
from src.data_admin import delete_record, load_clean_data, save_clean_data, upsert_record
from src.location_config import list_locations
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
        user = authenticate(username.strip(), password)
        if user is None:
            st.error("Invalid username or password")
        else:
            st.session_state["user"] = {"username": user.username, "role": user.role}
            st.rerun()
    st.stop()


if "user" not in st.session_state:
    login_gate()

user = st.session_state["user"]
if user["role"] not in {"admin"}:
    st.error("Admin role required for this dashboard.")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.stop()

locations = list_locations()
loc_names = {loc.name: loc.id for loc in locations}

st.title("Multi-Location Forecast Admin")
st.caption("Each location has independent database, model artifacts, and prediction outputs")

sidebar = st.sidebar
sidebar.markdown(f"**User:** {user['username']} ({user['role']})")
selected_name = sidebar.selectbox("Location", options=list(loc_names.keys()))
location_id = loc_names[selected_name]
if sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

model_path = model_file_for_location(location_id)
artifact_dir = artifact_dir_for_location(location_id)

if not model_path.exists():
    st.warning(f"Model not found for location '{location_id}'. Train this location first.")
    if st.button("Train this location", type="primary"):
        with st.spinner("Training..."):
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_backtest.py"), "--location", location_id], cwd=ROOT)
        if r.returncode == 0:
            st.success("Training completed")
            st.rerun()
        else:
            st.error("Training failed")
    st.stop()

predictor = VisitorPredictor(str(model_path))



def render_prediction():
    st.subheader(f"Prediction - {selected_name}")
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
                st.success("Saved")
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



t1, t2, t3 = st.tabs(["Prediction", "Metrics", "Data Management"])
with t1:
    render_prediction()
with t2:
    render_metrics()
with t3:
    render_data_ops()
