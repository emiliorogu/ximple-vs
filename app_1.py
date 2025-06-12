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
# PALETAS DE COLORES UNIFICADAS (AZUL)
# ----------------------------
COLOR_SEQ = px.colors.sequential.Blues  # Paleta azul para escalas continuas
COLOR_SEQ_DISCRETE = ['#1f77b4', '#3399e6', '#5dade2', '#85c1e9', '#aed6f1', '#d6eaf8']

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
# SIDEBAR MENU (add new sections)
# ----------------------------
menu = st.sidebar.selectbox(
    "Select a section",
    [
        "Home",
        "Demographics",
        "States and Gender",
        "Marketing & Acquisition",
        "Stage Progression",
        "Prediction & ML Model",
        "Full Dashboard",
        "Recommendations"
    ]
)

# ----------------------------
# SECTION: HOME (Landing Page)
# ----------------------------
if menu == "Home":
    st.title("Ximple: Empowering Financial Inclusion for Women Microentrepreneurs")
    st.markdown("""
    ## Introduction
    Financial inclusion remains a significant challenge for many microentrepreneurs in Latin America, especially women who face limited access to credit, formal employment, and digital tools. **Ximple** is a fintech platform created to address this issue by offering credit solutions tailored to the needs of underbanked women, known as *aliadas*, helping them launch and grow their businesses.

    ## Context
    Our objective is to support Ximple in optimizing its acquisition, onboarding, and loan delivery processes, ensuring that more women can successfully become active clients and access the financial support they need.

    Ximple provided two main datasets for this analysis:
    - **Enrollment Dataset:** Includes demographic details, registration progress, and onboarding status of aliadas.
    - **Channel Dataset:** Captures interactions across different acquisition channels, from initial contact to final loan delivery.

    ## Problem Statement
    Our analysis focused on solving two central questions:

    1. **How can Ximple improve the acquisition of aliadas through its various channels?**  
       We examined the performance of campaigns, sources, and communication channels to understand which strategies drive the highest reach and conversion into active users.

    2. **What factors influence the successful completion of the onboarding process?**  
       By analyzing demographic variables and registration behavior, we aimed to uncover the barriers that prevent aliadas from completing the enrollment process and accessing credit products.

    Through data-driven insights, our goal is to help Ximple increase the number of registered aliadas, improve their onboarding experience, and expand their access to loans and financial tools.
    """)

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
        color_continuous_scale=COLOR_SEQ,
        title='Average Loan Amount by State'
    ), use_container_width=True)

    st.plotly_chart(px.bar(
        df_loans.groupby('Assigned_State')['LoanAmount'].sum().reset_index(),
        x='LoanAmount', y='Assigned_State', orientation='h', color='LoanAmount',
        color_continuous_scale=COLOR_SEQ,
        title='Total Loan Amount by State'
    ), use_container_width=True)

    st.subheader("Completed Underwriting by Region and Gender (Bar Chart)")
    region_gender = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    region_gender = region_gender.groupby(['Region', 'Gender']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(region_gender, x='Region', y='Count', color='Gender',
                           color_discrete_sequence=COLOR_SEQ_DISCRETE,
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
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        hover_name="Region", zoom=4, height=600,
        title="Completed Underwriting by Region and Gender (Map)")
        .update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0}),
        use_container_width=True)

    st.subheader("Completed Underwriting by State and Gender (Bar Chart)")
    state_gender = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    state_gender = state_gender.groupby(['Assigned_State', 'Gender']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(state_gender, x='Assigned_State', y='Count', color='Gender',
                           color_discrete_sequence=COLOR_SEQ_DISCRETE,
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
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        hover_name="Assigned_State", zoom=4, height=600,
        title="Completed Underwriting by State and Gender (Map)")
        .update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0}),
        use_container_width=True)
    
# ----------------------------
# SECTION: MARKETING & ACQUISITION
# ----------------------------
elif menu == "Marketing & Acquisition":
    st.subheader("Loan Conversion Rates by Marketing Channel and Subchannel")

    # Replace nulls in tiene_prestamo with 0
    df_campaigns['tiene_prestamo'] = df_campaigns['tiene_prestamo'].fillna(0)

    # Filtrar marketing_channel con más de 100 usuarios
    channel_counts = df_campaigns['marketing_channel'].value_counts()
    valid_channels = channel_counts[channel_counts > 100].index
    df_channel = (
        df_campaigns[df_campaigns['marketing_channel'].isin(valid_channels)]
        .groupby('marketing_channel')['tiene_prestamo']
        .mean().reset_index().sort_values('tiene_prestamo', ascending=False)
    )
    df_channel['tiene_prestamo'] = df_channel['tiene_prestamo'] * 100

    st.plotly_chart(px.bar(
        df_channel,
        x='tiene_prestamo', y='marketing_channel', orientation='h',
        color='marketing_channel',
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        title='Conversion Rate by Marketing Channel (%)',
        labels={'tiene_prestamo': 'Loan Conversion Rate (%)', 'marketing_channel': 'Marketing Channel'}
    ), use_container_width=True)

    # Filtrar marketing_subchannel con más de 100 usuarios
    subchannel_counts = df_campaigns['marketing_subchannel'].value_counts()
    valid_subchannels = subchannel_counts[subchannel_counts > 100].index
    df_subchannel = (
        df_campaigns[df_campaigns['marketing_subchannel'].isin(valid_subchannels)]
        .groupby('marketing_subchannel')['tiene_prestamo']
        .mean().reset_index().sort_values('tiene_prestamo', ascending=False)
    )
    df_subchannel['tiene_prestamo'] = df_subchannel['tiene_prestamo'] * 100

    st.plotly_chart(px.bar(
        df_subchannel,
        x='tiene_prestamo', y='marketing_subchannel', orientation='h',
        color='marketing_subchannel',
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        title='Conversion Rate by Marketing Subchannel (%)',
        labels={'tiene_prestamo': 'Loan Conversion Rate (%)', 'marketing_subchannel': 'Marketing Subchannel'}
    ), use_container_width=True)

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
                 color_continuous_scale=COLOR_SEQ)
    st.plotly_chart(fig, use_container_width=True)

    # Add UOS Approved vs Rejected
    st.subheader("UOS Aprobado vs. Rechazado")
    uos_counts = df_funnel['Underwriting overall status'].value_counts().reset_index()
    uos_counts.columns = ['Underwriting Status', 'Count']
    fig_uos = px.bar(uos_counts, x='Underwriting Status', y='Count',
                     color='Underwriting Status',
                     color_discrete_sequence=COLOR_SEQ_DISCRETE,
                     title='UOS Aprobado vs. Rechazado'
    )
    st.plotly_chart(fig_uos, use_container_width=True)

# ----------------------------
# SECTION: PREDICTION & ML MODEL
# ----------------------------
elif menu == "Prediction & ML Model":
    st.subheader("Predicted Marketing Channels and Acquisition Paths")

    col1, col2 = st.columns(2)
    with col1:
        channel_data = df_campaigns['marketing_channel'].value_counts().reset_index()
        channel_data.columns = ['marketing_channel', 'Count']
        st.plotly_chart(px.bar(
            channel_data.sort_values(by='Count'),
            y='marketing_channel', x='Count', orientation='h',
            color='marketing_channel',
            color_discrete_sequence=COLOR_SEQ_DISCRETE,
            title='Top Marketing Channels by Predicted Loan Conversions'
        ), use_container_width=True)

    with col2:
        subchannel_data = df_campaigns['marketing_subchannel'].value_counts().reset_index()
        subchannel_data.columns = ['marketing_subchannel', 'Count']
        st.plotly_chart(px.bar(
            subchannel_data.sort_values(by='Count'),
            y='marketing_subchannel', x='Count', orientation='h',
            color='marketing_subchannel',
            color_discrete_sequence=COLOR_SEQ_DISCRETE,
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
            title='Predicted Loan Conversions by Day of Week',
            color_discrete_sequence=COLOR_SEQ_DISCRETE
        )
        fig.update_layout(xaxis_title='Day of Week', yaxis_title='Predicted Conversions')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weekday data available in 'dia_semana_entrada'.")

    st.markdown("---")
    st.write("Note: The Random Forest model was trained on features such as source, medium, and entry day. The classification threshold was manually tuned to improve precision.")

# ----------------------------
# SECTION: FULL DASHBOARD (ALL-IN-ONE)
# ----------------------------
elif menu == "Full Dashboard":
    st.title("Full Ximple VS Dashboard")

    # --- Demographics ---
    st.header("Demographics")
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
        width=800,
        title='Age Distribution: All users vs users with completed underwriting'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # --- States and Gender ---
    st.header("States and Gender")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(
            df_loans.groupby('Assigned_State')['LoanAmount'].mean().reset_index(),
            y='Assigned_State', x='LoanAmount', orientation='h', color='LoanAmount',
            color_continuous_scale=COLOR_SEQ,
            title='Average Loan Amount by State'
        ), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(
            df_loans.groupby('Assigned_State')['LoanAmount'].sum().reset_index(),
            x='LoanAmount', y='Assigned_State', orientation='h', color='LoanAmount',
            color_continuous_scale=COLOR_SEQ,
            title='Total Loan Amount by State'
        ), use_container_width=True)

    st.subheader("Completed Underwriting by Region and Gender (Bar Chart)")
    region_gender = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    region_gender = region_gender.groupby(['Region', 'Gender']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(region_gender, x='Region', y='Count', color='Gender',
                           color_discrete_sequence=COLOR_SEQ_DISCRETE,
                           title="Completed Underwriting by Region and Gender"), use_container_width=True)

    st.subheader("States by Region (Map)")
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
    state_region_map = df_funnel[['Assigned_State', 'Region']].dropna().drop_duplicates()
    state_region_map['Lat'] = state_region_map['Assigned_State'].map(lambda x: state_coords.get(x, (None, None))[0])
    state_region_map['Lon'] = state_region_map['Assigned_State'].map(lambda x: state_coords.get(x, (None, None))[1])
    state_region_map = state_region_map.dropna(subset=['Lat', 'Lon'])
    fig = px.scatter_mapbox(
        state_region_map,
        lat="Lat",
        lon="Lon",
        color="Region",
        hover_name="Assigned_State",
        size_max=15,
        zoom=4,
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        title="States colored by Region"
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Completed Underwriting by State and Gender (Bar Chart)")
    state_gender = df_funnel[df_funnel['Underwriting overall status'] == 'COMPLETED']
    state_gender = state_gender.groupby(['Assigned_State', 'Gender']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(state_gender, x='Assigned_State', y='Count', color='Gender',
                           color_discrete_sequence=COLOR_SEQ_DISCRETE,
                           title="Completed Underwriting by State and Gender"), use_container_width=True)

    st.subheader("Completed Underwriting by State and Gender (Map)")
    state_gender['Lat'] = state_gender['Assigned_State'].map(lambda x: state_coords.get(x, (None, None))[0])
    state_gender['Lon'] = state_gender['Assigned_State'].map(lambda x: state_coords.get(x, (None, None))[1])
    state_gender = state_gender.dropna()
    st.plotly_chart(px.scatter_map(
        state_gender, lat="Lat", lon="Lon", size="Count", color="Gender",
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        hover_name="Assigned_State", zoom=4, height=600,
        title="Completed Underwriting by State and Gender (Map)")
        .update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0}),
        use_container_width=True)

    # --- Marketing & Acquisition ---
    st.header("Marketing & Acquisition")

    # Conversion rate by marketing_channel (más de 100 usuarios)
    channel_counts = df_campaigns['marketing_channel'].value_counts()
    valid_channels = channel_counts[channel_counts > 100].index
    df_channel = (
        df_campaigns[df_campaigns['marketing_channel'].isin(valid_channels)]
        .groupby('marketing_channel')['tiene_prestamo']
        .mean().reset_index().sort_values('tiene_prestamo', ascending=False)
    )
    df_channel['tiene_prestamo'] = df_channel['tiene_prestamo'] * 100

    st.plotly_chart(px.bar(
        df_channel,
        x='tiene_prestamo', y='marketing_channel', orientation='h',
        color='marketing_channel',
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        title='Conversion Rate by Marketing Channel (%)',
        labels={'tiene_prestamo': 'Loan Conversion Rate (%)', 'marketing_channel': 'Marketing Channel'}
    ), use_container_width=True)

    # Conversion rate by marketing_subchannel (más de 100 usuarios)
    subchannel_counts = df_campaigns['marketing_subchannel'].value_counts()
    valid_subchannels = subchannel_counts[subchannel_counts > 100].index
    df_subchannel = (
        df_campaigns[df_campaigns['marketing_subchannel'].isin(valid_subchannels)]
        .groupby('marketing_subchannel')['tiene_prestamo']
        .mean().reset_index().sort_values('tiene_prestamo', ascending=False)
    )
    df_subchannel['tiene_prestamo'] = df_subchannel['tiene_prestamo'] * 100

    st.plotly_chart(px.bar(
        df_subchannel,
        x='tiene_prestamo', y='marketing_subchannel', orientation='h',
        color='marketing_subchannel',
        color_discrete_sequence=COLOR_SEQ_DISCRETE,
        title='Conversion Rate by Marketing Subchannel (%)',
        labels={'tiene_prestamo': 'Loan Conversion Rate (%)', 'marketing_subchannel': 'Marketing Subchannel'}
    ), use_container_width=True)

    # --- Stage Progression ---
    st.header("Stage Progression")
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
                 color_continuous_scale=COLOR_SEQ)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("UOS Aprobado vs. Rechazado")
    uos_counts = df_funnel['Underwriting overall status'].value_counts().reset_index()
    uos_counts.columns = ['Underwriting Status', 'Count']
    fig_uos = px.bar(uos_counts, x='Underwriting Status', y='Count',
                     color='Underwriting Status',
                     color_discrete_sequence=COLOR_SEQ_DISCRETE,
                     title='UOS Aprobado vs. Rechazado'
    )
    st.plotly_chart(fig_uos, use_container_width=True)

    # --- Prediction & ML Model ---
    st.header("Prediction & ML Model")
    col1, col2 = st.columns(2)
    with col1:
        channel_data = df_campaigns['marketing_channel'].value_counts().reset_index()
        channel_data.columns = ['marketing_channel', 'Count']
        st.plotly_chart(px.bar(
            channel_data.sort_values(by='Count'),
            y='marketing_channel', x='Count', orientation='h',
            color='marketing_channel',
            color_discrete_sequence=COLOR_SEQ_DISCRETE,
            title='Top Marketing Channels by Predicted Loan Conversions'
        ), use_container_width=True)
    with col2:
        subchannel_data = df_campaigns['marketing_subchannel'].value_counts().reset_index()
        subchannel_data.columns = ['marketing_subchannel', 'Count']
        st.plotly_chart(px.bar(
            subchannel_data.sort_values(by='Count'),
            y='marketing_subchannel', x='Count', orientation='h',
            color='marketing_subchannel',
            color_discrete_sequence=COLOR_SEQ_DISCRETE,
            title='Top 10 Marketing Subchannels by Predicted Loan Conversions'
        ), use_container_width=True)

    st.markdown("### Predicted Loan Conversions by Day of Week")
    if 'dia_semana_entrada' in df_campaigns.columns and not df_campaigns['dia_semana_entrada'].isnull().all():
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
            title='Predicted Loan Conversions by Day of Week',
            color_discrete_sequence=COLOR_SEQ_DISCRETE
        )
        fig.update_layout(xaxis_title='Day of Week', yaxis_title='Predicted Conversions')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weekday data available in 'dia_semana_entrada'.")

# ----------------------------
# SECTION: RECOMMENDATIONS (Ending)
# ----------------------------
elif menu == "Recommendations":
    st.title("Conclusions & Recommendations")
    st.markdown("""
    ## Conclusions

    This project provided valuable insights into the behavior and needs of Ximple’s aliadas throughout the acquisition and onboarding journey. By analyzing the data, we were able to identify key patterns that impact user engagement, registration completion, and loan usage. These findings not only revealed areas where the process can be streamlined but also highlighted the importance of using the right channels at the right stages. Leveraging these insights allows Ximple to make more informed, data-driven decisions to improve reach, increase conversion, and better support the women they aim to empower.

    ## Recommendations

    1. **Use Paid Social/Facebook campaigns for initial acquisition.**  
       These channels are ideal for raising awareness and attracting new users, but should be supported with stronger follow-up strategies to ensure engagement and activation.

    2. **Prioritize WhatsApp and direct communication as activation channels.**  
       These methods show the highest conversion rates. Focus efforts on these channels to guide aliadas through onboarding and toward effective use of loans.

    3. **Integrate the machine learning model into operational workflows.**  
       Implement the model to identify low-probability loans users early, allowing the team to focus resources on those with higher potential for conversion and loan usage.

    4. **Simplify the onboarding process and provide timely support at key drop-off points.**  
       Offering automated reminders, and personalized assistance at critical steps can significantly improve completion and activation rates. This assistance could come from a button connected to the WhatsApp Channel, due to its higher conversion rate, and lead to a chatbot, which can provide personalized assistance and accessibility to the Aliadas.
    """)

