import streamlit as st

st.page_link("pages/01_Finance__Dashboard.py", label="Open Finance →")
st.page_link("pages/10_Flips__Search.py", label="Open Flips →")
st.page_link("pages/20_Autos__Search.py", label="Open Autos →")
st.page_link("pages/02_Finance__Single_Ticker.py", label="Single Ticker →")
st.page_link("pages/03_Finance__Scanner.py", label="Scanner →")



st.set_page_config(page_title="Ed's Dashboards", layout="wide")
st.title("Dashboards")
st.caption("Landing page for Finance, House Flips, and Autos/Trucks")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📈 Finance")
    st.write("Portfolio, signals, charts")
    st.page_link("pages/01_Finance__Dashboard.py", label="Open Finance →")

with c2:
    st.subheader("🏠 House Flips")
    st.write("Deal finder, comps, ARV/MAO")
    st.page_link("pages/10_Flips__Search.py", label="Open Flips →")

with c3:
    st.subheader("🚗 Autos & Trucks")
    st.write("Vehicle deal finder, VIN decode, comps")
    st.page_link("pages/20_Autos__Search.py", label="Open Autos →")
