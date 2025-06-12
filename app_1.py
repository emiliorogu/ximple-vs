# ximple_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Ximple VS Analytics Dashboard", layout="wide")
st.title("Ximple VS Analytics Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data

def load_data():
    df_campaigns = pd.read_csv("Campañas - Aliadas Campaigns.csv")
    df_funnel = pd.read_csv("df_dropped.csv")
    df_eq1 = pd.read_csv("BaseEQ1.csv", dtype=str)
    df_loans = pd.read_csv("Loans_Ximple.csv")
    return df_campaigns, df_funnel, df_eq1, df_loans

# Load data once at startup
with st.spinner("Loading data..."):
    df_campaigns, df_funnel, df_eq1, df_loans = load_data()

# Add Assigned_State to df_loans by merging with df_funnel
if 'Assigned_State' not in df_loans.columns:
    df_loans = pd.merge(
        df_loans,
        df_funnel[['External account H', 'Assigned_State']],
        left_on='Ally ID',
        right_on='External account H',
        how='left'
    )

# Ensure 'dia_semana_entrada' exists in df_campaigns based on fecha_prestamo
if 'fecha_prestamo' in df_campaigns.columns:
    df_campaigns['fecha_prestamo'] = pd.to_numeric(df_campaigns['fecha_prestamo'], errors='coerce')
    df_campaigns['fecha_prestamo'] = df_campaigns['fecha_prestamo'].dropna().astype(int).astype(str)
    df_campaigns['fecha_prestamo'] = pd.to_datetime(df_campaigns['fecha_prestamo'], format='%Y%m%d', errors='coerce')
    df_campaigns['dia_semana_entrada'] = df_campaigns['fecha_prestamo'].dt.dayofweek

# Create acquisition_path if missing
if 'acquisition_path' not in df_campaigns.columns and 'marketing_channel' in df_campaigns.columns and 'marketing_subchannel' in df_campaigns.columns:
    df_campaigns['acquisition_path'] = (
        df_campaigns['marketing_channel'].astype(str).str.lower().str.strip() + " | " +
        df_campaigns['marketing_subchannel'].astype(str).str.lower().str.strip()
    )

# Ensure 'Underwriting overall status' exists in df_campaigns by merging from df_funnel if possible
# Try to identify the best matching key to merge on
merge_keys_df_campaigns = ['Ally ID', 'RecipientID', 'user_id']
merge_key_campaign = next((col for col in merge_keys_df_campaigns if col in df_campaigns.columns), None)

if merge_key_campaign and 'External account H' in df_funnel.columns and 'Underwriting overall status' in df_funnel.columns:
    df_campaigns[merge_key_campaign] = df_campaigns[merge_key_campaign].astype(str)
    df_funnel['External account H'] = df_funnel['External account H'].astype(str)

    df_campaigns = pd.merge(
        df_campaigns,
        df_funnel[['External account H', 'Underwriting overall status']],
        left_on=merge_key_campaign,
        right_on='External account H',
        how='left'
    )
else:
    st.warning("Missing required columns to merge 'Underwriting overall status'.")

# ----------------------------
# SIDEBAR MENU
# ----------------------------
menu = st.sidebar.selectbox(
    "Select a section",
    [
        "Demographics",
        "States and Gender",
        "Marketing & Acquisition",
        "Stage Progression",
        "Prediction & ML Model"
    ]
)

# ----------------------------
# SECTION: DEMOGRAPHICS
# ----------------------------
if menu == "Demographics":
    st.subheader("Age Distribution")

    df_all = df_funnel['AGE'].value_counts().reset_index()
    df_all.columns = ['AGE', 'Count']
    df_all['Dataset'] = 'All users'

    df_uos = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    df_uos = df_uos[df_uos['AGE'] >= 18]
    df_uos_age = df_uos['AGE'].value_counts().reset_index()
    df_uos_age.columns = ['AGE', 'Count']
    df_uos_age['Dataset'] = 'Completed underwriting'

    age_dist = pd.concat([df_all, df_uos_age])

    chart = alt.Chart(age_dist).mark_bar().encode(
        x=alt.X('AGE:O', title='Age'),
        y=alt.Y('Count:Q', title='Frequency'),
        color='Dataset:N',
        tooltip=['AGE', 'Count', 'Dataset']
    ).properties(
        width=1000,
        title='Age Distribution: All users vs users with completed underwriting'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# ----------------------------
# SECTION: STATES AND GENDER
# ----------------------------
elif menu == "States and Gender":
    st.subheader("Distribution by State and Region")

    st.plotly_chart(px.bar(
        df_loans.groupby('Assigned_State')['LoanAmount'].mean().reset_index(),
        y='Assigned_State', x='LoanAmount', orientation='h', color='LoanAmount',
        color_continuous_scale='YlGnBu',
        title='Average Loan Amount by State'
    ), use_container_width=True)

    st.plotly_chart(px.bar(
        df_loans.groupby('Assigned_State')['LoanAmount'].sum().reset_index(),
        x='LoanAmount', y='Assigned_State', orientation='h', color='LoanAmount',
        title='Total Loan Amount by State'
    ), use_container_width=True)

    st.subheader("Completed Underwriting by Region and Gender (Bar Chart)")
    region_gender = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    region_gender = region_gender.groupby(['Region', 'Gender']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(region_gender, x='Region', y='Count', color='Gender',
                           title="Completed Underwriting by Region and Gender"), use_container_width=True)

    st.subheader("Completed Underwriting by Region and Gender (Map)")
    region_coords = {
        'Centro': (19.4326, -99.1332), 'Centro-Oriental': (20.0911, -98.7624), 'Sureste': (17.0732, -96.7266),
        'Centro-Norte': (22.1565, -100.9855), 'Centro-Occidente': (20.6597, -103.3496), 'Norte': (27.0587, -101.7068),
        'Sur': (16.7569, -93.1292), 'Noroeste': (29.2972, -110.3309), 'Este': (19.1738, -96.1342),
        'Occidente': (21.0190, -101.2574), 'Centro-Sur': (18.6813, -99.1013)
    }
    region_gender['Lat'] = region_gender['Region'].map(lambda x: region_coords.get(x, (None, None))[0])
    region_gender['Lon'] = region_gender['Region'].map(lambda x: region_coords.get(x, (None, None))[1])
    region_gender = region_gender.dropna()
    st.plotly_chart(px.scatter_map(
        region_gender, lat="Lat", lon="Lon", size="Count", color="Gender",
        hover_name="Region", zoom=4, height=600,
        title="Completed Underwriting by Region and Gender (Map)")
        .update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0}),
        use_container_width=True)

    st.subheader("Completed Underwriting by State and Gender (Bar Chart)")
    state_gender = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    state_gender = state_gender.groupby(['Assigned_State', 'Gender']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(state_gender, x='Assigned_State', y='Count', color='Gender',
                           title="Completed Underwriting by State and Gender"), use_container_width=True)

    st.subheader("Completed Underwriting by State and Gender (Map)")
    state_coords = {
        'Aguascalientes': (21.8853, -102.2916), 'Baja California': (30.8406, -115.2838),
        'Baja California Sur': (26.0444, -111.6661), 'Campeche': (19.8301, -90.5349),
        'Chiapas': (16.7569, -93.1292), 'Chihuahua': (28.6329, -106.0691),
        'Ciudad de México': (19.4326, -99.1332), 'Coahuila': (27.0587, -101.7068),
        'Colima': (19.1223, -103.8840), 'Durango': (24.5593, -104.6580), 'Guanajuato': (21.0190, -101.2574),
        'Guerrero': (17.4392, -99.5451), 'Hidalgo': (20.0911, -98.7624), 'Jalisco': (20.6597, -103.3496),
        'México (Estado de México)': (19.3587, -99.8707), 'Michoacán': (19.1538, -101.8831),
        'Morelos': (18.6813, -99.1013), 'Nayarit': (21.7514, -104.8455), 'Nuevo León': (25.5922, -99.9962),
        'Oaxaca': (17.0732, -96.7266), 'Puebla': (19.0414, -98.2063), 'Querétaro': (20.5888, -100.3899),
        'Quintana Roo': (19.1817, -88.4791), 'San Luis Potosí': (22.1565, -100.9855), 'Sinaloa': (25.1721, -107.4795),
        'Sonora': (29.2972, -110.3309), 'Tabasco': (17.8409, -92.6189), 'Tamaulipas': (23.7369, -99.1411),
        'Tlaxcala': (19.3139, -98.2400), 'Veracruz': (19.1738, -96.1342), 'Yucatán': (20.7099, -89.0943),
        'Zacatecas': (22.7709, -102.5832)
    }
    state_gender['Lat'] = state_gender['Assigned_State'].map(lambda x: state_coords.get(x, (None, None))[0])
    state_gender['Lon'] = state_gender['Assigned_State'].map(lambda x: state_coords.get(x, (None, None))[1])
    state_gender = state_gender.dropna()
    st.plotly_chart(px.scatter_map(
        state_gender, lat="Lat", lon="Lon", size="Count", color="Gender",
        hover_name="Assigned_State", zoom=4, height=600,
        title="Completed Underwriting by State and Gender (Map)")
        .update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0}),
        use_container_width=True)

# ----------------------------
# SECTION: MARKETING & ACQUISITION
# ----------------------------
if menu == "Marketing & Acquisition":
    st.subheader("Marketing Channels and Acquisition Paths")

    col1, col2 = st.columns(2)
    with col1:
        channel_data = df_campaigns['marketing_channel'].value_counts().reset_index()
        channel_data.columns = ['marketing_channel', 'Count']
        st.plotly_chart(px.bar(
            channel_data.sort_values(by='Count'),
            y='marketing_channel', x='Count', orientation='h',
            title='Top Marketing Channels by Predicted Loan Conversions'
        ), use_container_width=True)

    with col2:
        subchannel_data = df_campaigns['marketing_subchannel'].value_counts().reset_index()
        subchannel_data.columns = ['marketing_subchannel', 'Count']
        st.plotly_chart(px.bar(
            subchannel_data.sort_values(by='Count'),
            y='marketing_subchannel', x='Count', orientation='h',
            title='Top 10 Marketing Subchannels by Predicted Loan Conversions'
        ), use_container_width=True)

    st.markdown("### Predicted Loan Conversions by Day of Week")
    if 'dia_semana_entrada' in df_campaigns.columns and not df_campaigns['dia_semana_entrada'].isnull().all():
        st.write(df_campaigns['dia_semana_entrada'].value_counts())
        df_day = df_campaigns.copy()
        df_day = df_day[df_day['dia_semana_entrada'].notnull()]
        df_day['dia_semana_entrada'] = df_day['dia_semana_entrada'].astype(int)
        df_day = df_day.groupby('dia_semana_entrada').size().reset_index(name='predicted_converters')
        df_day['Day'] = df_day['dia_semana_entrada'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        df_day = df_day.sort_values('dia_semana_entrada')
        fig = px.line(
            df_day,
            x='Day',
            y='predicted_converters',
            markers=True,
            title='Predicted Loan Conversions by Day of Week'
        )
        fig.update_layout(xaxis_title='Day of Week', yaxis_title='Predicted Conversions')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weekday data available in 'dia_semana_entrada'.")

# ----------------------------
# SECTION: STAGE PROGRESSION
# ----------------------------
elif menu == "Stage Progression":
    st.subheader("Completed Stages per User")

    completed_df_filtered = df_funnel.copy()
    cols = [c for c in completed_df_filtered.columns if 'status' in c.lower() and 'overall' not in c.lower()]
    completed_df_filtered = pd.DataFrame({
        'Status Column': cols,
        'COMPLETED Count': [completed_df_filtered[c].eq('COMPLETED').sum() for c in cols]
    }).sort_values(by='COMPLETED Count', ascending=False)

    fig = px.bar(completed_df_filtered,
                 x='COMPLETED Count', y='Status Column', orientation='h',
                 title='Completed Counts Progression for Each Status Column',
                 color='COMPLETED Count',
                 color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

    # Add UOS Approved vs Rejected
    st.subheader("UOS Aprobado vs. Rechazado")
    uos_counts = df_funnel['Underwriting overall status'].value_counts().reset_index()
    uos_counts.columns = ['Underwriting Status', 'Count']
    fig_uos = px.bar(uos_counts, x='Underwriting Status', y='Count',
                     color='Underwriting Status',
                     title='UOS Aprobado vs. Rechazado')
    st.plotly_chart(fig_uos, use_container_width=True)

# ----------------------------
# SECTION: PREDICTION & ML MODEL
# ----------------------------
elif menu == "Prediction & ML Model":
    st.subheader("Model Performance and Predictions")

    df_top_paths = (
        df_campaigns.groupby('acquisition_path')['tiene_prestamo']
        .mean().reset_index().nlargest(10, 'tiene_prestamo')
    )
    df_top_paths['tiene_prestamo'] = df_top_paths['tiene_prestamo'] * 100  # percent

    st.plotly_chart(px.bar(
        df_top_paths,
        x='tiene_prestamo', y='acquisition_path', orientation='h',
        title='Top Acquisition Paths by Loan Conversion Rate (%)',
        labels={'tiene_prestamo': 'Loan Conversion Rate (%)', 'acquisition_path': 'Acquisition Path'}
    ), use_container_width=True)

    st.markdown("---")
    st.write("Note: The Random Forest model was trained on features such as source, medium, and entry day. The classification threshold was manually tuned to improve precision.")