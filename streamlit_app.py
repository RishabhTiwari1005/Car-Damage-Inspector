from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app import db
from app.config import DAMAGE_WEIGHTS, PARTS_WEIGHTS, STATUSES
from app.detector import Detector
from app.pipeline import run_inspection
from app.report import build_report, export_pdf

st.set_page_config(page_title="Car Damage Inspector", layout="wide")
db.init_db()


@st.cache_resource
def load_detector():
    return Detector(PARTS_WEIGHTS, DAMAGE_WEIGHTS)


def show_report(report):
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall severity", report["overall_severity"])
    c2.metric("Findings", len(report["items"]))
    c3.metric(f"Est. cost ({report['currency']})", f"{report['total_low']:.0f} - {report['total_high']:.0f}")
    if report["items"]:
        df = pd.DataFrame(report["items"])
        df["images"] = df["images"].apply(lambda x: ", ".join(x))
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No damage detected.")
    st.caption(report["disclaimer"])


def show_images(images):
    cols = st.columns(min(len(images), 3) or 1)
    for i, (name, data) in enumerate(images.items()):
        cols[i % len(cols)].image(data, caption=name, use_container_width=True)


new_tab, history_tab = st.tabs(["New inspection", "History"])

with new_tab:
    st.subheader("Upload photos of one car")
    car_label = st.text_input("Car label (plate, VIN or bay number)", value="")
    files = st.file_uploader("Photos from different angles", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Run inspection", type="primary", disabled=not files):
        images = []
        for f in files:
            img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                images.append((f.name, img))
        try:
            with st.spinner("Detecting parts and damage..."):
                findings, annotated = run_inspection(load_detector(), images)
            st.session_state["result"] = {
                "car_label": car_label or "unlabeled",
                "report": build_report(findings),
                "annotated": annotated,
                "saved_id": None,
            }
        except FileNotFoundError as e:
            st.error(str(e))

    result = st.session_state.get("result")
    if result:
        show_report(result["report"])
        show_images(result["annotated"])
        c1, c2 = st.columns(2)
        if result["saved_id"] is None:
            if c1.button("Save to history"):
                result["saved_id"] = db.save_inspection(result["car_label"], result["report"], result["annotated"])
                st.rerun()
        else:
            c1.success(f"Saved as inspection #{result['saved_id']}")
        c2.download_button(
            "Download PDF report",
            export_pdf(result["car_label"], result["report"], result["annotated"]),
            file_name=f"report_{result['car_label']}.pdf",
            mime="application/pdf",
        )

with history_tab:
    rows = db.list_inspections()
    if not rows:
        st.info("No saved inspections yet.")
    else:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Open inspection", [r["id"] for r in rows], format_func=lambda i: f"#{i}")
        record = db.get_inspection(chosen)
        st.markdown(f"**{record['car_label']}** | {record['created_at']} | status: **{record['status']}**")
        new_status = st.selectbox("Set status", STATUSES, index=STATUSES.index(record["status"]))
        if st.button("Update status"):
            db.set_status(chosen, new_status)
            st.rerun()
        show_report(record["report"])
        images = {Path(p).name: Path(p).read_bytes() for p in record["image_files"]}
        show_images(images)
        st.download_button(
            "Download PDF report",
            export_pdf(record["car_label"], record["report"], images, record["created_at"]),
            file_name=f"report_{chosen}.pdf",
            mime="application/pdf",
            key=f"pdf_{chosen}",
        )
