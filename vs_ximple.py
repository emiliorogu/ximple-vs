# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

# --- Data Loading ---
def load_data():
    df_campaigns = pd.read_csv("Campañas - Aliadas Campaigns.csv")
    df_funnel = pd.read_csv("df_dropped.csv")
    df = pd.read_excel("BaseEQ1.xlsx", engine="openpyxl")
    df_loans = pd.read_excel("Loans Ximple.xlsx", engine="openpyxl")
    return df_campaigns, df_funnel, df, df_loans

# --- Data Cleaning ---
def clean_na(df):
    df.replace('N A', pd.NA, inplace=True)
    return df

def drop_columns(df, columns):
    return df.drop(columns=columns, errors='ignore')

# --- Main Execution ---
if __name__ == "__main__":
    # Load and clean data
    df_campaigns, df_funnel, df, df_loans = load_data()
    df = clean_na(df)
    df_campaigns = clean_na(df_campaigns)

    # Filter only data from December 2024 onward
    df['Updated at'] = pd.to_datetime(df['Updated at'])
    new_df = df[df['Updated at'] >= '2024-12-01']

    # Drop irrelevant columns
    columns_to_drop = ["Phone H", "Mobile Score", "Column 42"]
    df_dropped = drop_columns(new_df, columns_to_drop)

    # Filter adults (18+)
    df_dropped = df_dropped[df_dropped["AGE"] >= 18]

    # Explore unique state values
    print("States value counts:")
    print(df_dropped['State'].value_counts())
    print("All states:")
    print(df_dropped['State'].unique())

    # ZIP code mappings by state
    zip_codes = {
        "Aguascalientes": [20],
        "Baja California": [21, 22],
        "Baja California Sur": [23],
        "Campeche": [24],
        "Coahuila": [25, 26, 27],
        "Colima": [28],
        "Chiapas": [29, 30],
        "Chihuahua": [31, 32, 33],
        "Ciudad de México": list(range(1, 17)),
        "Durango": [34, 35],
        "Guanajuato": [36, 37, 38],
        "Guerrero": [39, 40, 41],
        "Hidalgo": [42, 43],
        "Jalisco": [44, 45, 46, 47, 48, 49],
        "México (Estado de México)": [50, 51, 52, 53, 54, 55, 56, 57],
        "Michoacán": [58, 59, 60, 61],
        "Morelos": [62],
        "Nayarit": [63],
        "Nuevo León": [64, 65, 66, 67],
        "Oaxaca": [68, 69, 70, 71],
        "Puebla": [72, 73, 74, 75],
        "Querétaro": [76],
        "Quintana Roo": [77],
        "San Luis Potosí": [78, 79],
        "Sinaloa": [80, 81, 82],
        "Sonora": [83, 84, 85],
        "Tabasco": [86],
        "Tamaulipas": [87, 88, 89],
        "Tlaxcala": [90],
        "Veracruz": [91, 92, 93, 94, 95, 96],
        "Yucatán": [97],
        "Zacatecas": [98, 99]
    }

    def assign_zip_code_key(zip_code, zip_codes):
        try:
            zip_code_str = str(int(zip_code))
            first_two_digits = int(zip_code_str[:2]) if len(zip_code_str) >= 2 else None
            for state, codes in zip_codes.items():
                if first_two_digits in codes:
                    return state
            return "Unknown"
        except (ValueError, TypeError):
            return np.nan

    # Apply zip code mapping
    df_dropped['Assigned_State'] = df_dropped['Zip Code'].apply(lambda x: assign_zip_code_key(x, zip_codes))
    df_dropped = df_dropped.drop(columns=['Zip Code', 'State'], errors='ignore')

    # Regional mapping
    region_mapping = {
        'Aguascalientes': 'Centro-Norte',
        'Baja California': 'Noroeste',
        'Baja California Sur': 'Noroeste',
        'Campeche': 'Sureste',
        'Coahuila': 'Norte',
        'Colima': 'Occidente',
        'Chiapas': 'Sureste',
        'Chihuahua': 'Norte',
        'Ciudad de México': 'Centro',
        'Durango': 'Norte',
        'Guanajuato': 'Centro-Occidente',
        'Guerrero': 'Sur',
        'Hidalgo': 'Centro-Oriental',
        'Jalisco': 'Occidente',
        'México (Estado de México)': 'Centro',
        'Michoacán': 'Occidente',
        'Morelos': 'Centro-Sur',
        'Nayarit': 'Occidente',
        'Nuevo León': 'Norte',
        'Oaxaca': 'Sur',
        'Puebla': 'Centro-Oriental',
        'Querétaro': 'Centro-Occidente',
        'Quintana Roo': 'Sureste',
        'San Luis Potosí': 'Centro-Norte',
        'Sinaloa': 'Noroeste',
        'Sonora': 'Noroeste',
        'Tabasco': 'Sureste',
        'Tamaulipas': 'Norte',
        'Tlaxcala': 'Centro',
        'Veracruz': 'Este',
        'Yucatán': 'Sureste',
        'Zacatecas': 'Norte'
    }
    df_dropped['Region'] = df_dropped['Assigned_State'].map(region_mapping)

    """**Data Transformation**"""

    df_combined = pd.merge(df_dropped, df_loans, left_on='External account H', right_on='Ally ID', how='left')
    df_combined = df_combined.dropna(subset=['External account H'])

    UOScompleted_users = df_dropped[df_dropped["Underwriting overall status"] == "COMPLETED"]
    print(UOScompleted_users.shape)

    state_counts = UOScompleted_users['Assigned_State'].value_counts()
    gender_counts = UOScompleted_users['Gender'].value_counts()
    df_dropped = df_dropped[df_dropped["AGE"] >= 18]

    # Define age bins and labels
    age_bins = [18, 24, 34, 44, 54, 64, float('inf')]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']

    # Cut the 'AGE' column into the defined bins
    UOScompleted_users['Age_Group'] = pd.cut(UOScompleted_users['AGE'], bins=age_bins, labels=age_labels, right=True)

    # Calculate the counts for each age group
    age_counts = UOScompleted_users['Age_Group'].value_counts().sort_index()

    df_uos_completed = df_dropped[df_dropped['Underwriting overall status'] == 'COMPLETED']
    df_CAdropped = df_dropped.dropna(subset=['Campaign identifier'])

    """# **Demographics**

    - Age range/ staistics (histogram)
    """

    # prompt: do age distribution df_dropped vs. df_uos_completed bar graph with altair

    import pandas as pd
    import altair as alt

    # Calculate the age distribution for both dataframes
    df_dropped_age_dist = df_dropped['AGE'].value_counts().reset_index()
    df_dropped_age_dist.columns = ['AGE', 'Count']
    df_dropped_age_dist['Dataset'] = 'All users'

    df_uos_completed_age_dist = df_uos_completed['AGE'].value_counts().reset_index()
    df_uos_completed_age_dist.columns = ['AGE', 'Count']
    df_uos_completed_age_dist['Dataset'] = 'Users with completed underwriting'

    # Combine the age distributions
    age_dist_combined = pd.concat([df_dropped_age_dist, df_uos_completed_age_dist])

    # Create the Altair bar chart
    chart = alt.Chart(age_dist_combined).mark_bar().encode(
        x=alt.X('AGE:O', title='Age'),  # Use ordinal for discrete ages
        y=alt.Y('Count:Q', title='Frequency'),
        color='Dataset:N',
        tooltip=['AGE', 'Count', 'Dataset']
    ).properties(
        title='Age Distribution: All users vs users with completed underwriting',
        width=800,
    ).interactive()

    chart.display()

    def descriptive_stats(df, column):
        if column not in df.columns:
            print(f"Error: Column '{column}' not found in the DataFrame.")
            return

        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Error: Column '{column}' is not numeric. Descriptive statistics are not applicable.")
            return

        mean = df[column].mean()
        std = df[column].std()
        q1 = df[column].quantile(0.25)
        median = df[column].median()
        q3 = df[column].quantile(0.75)

        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Standard Deviation', 'Q1', 'Median (Q2)', 'Q3'],
            column: [mean, std, q1, median, q3]
        })

        print(f"Descriptive Statistics for '{column}':")
        print(stats_df)

    descriptive_stats(df_uos_completed, 'AGE')
    descriptive_stats(df_uos_completed, 'FICO')
    descriptive_stats(df_uos_completed, 'Credit line')

    """- Underwriting complete by region (map)"""

    import pandas as pd
    import matplotlib.pyplot as plt

    state_gender_counts = df_uos_completed.groupby(['Assigned_State', 'Gender'])['External account H'].count().unstack(fill_value=0)

    print(state_gender_counts)

    ax = state_gender_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Completed Underwriting by State and Gender')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Gender')
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container, label_type='edge')

    plt.show()

    """- Average Loan Amount per Assigned State (bar graph)"""

    state_coordinates = {
        'México (Estado de México)': (19.3587, -99.8707),
        'Puebla': (19.0414, -98.2063),
        'Chiapas': (16.7569, -93.1292),
        'Ciudad de México': (19.4326, -99.1332),
        'San Luis Potosí': (22.1565, -100.9855),
        'Guanajuato': (21.0190, -101.2574),
        'Tlaxcala': (19.3139, -98.2400),
        'Quintana Roo': (19.1817, -88.4791),
        'Querétaro': (20.5888, -100.3899),
        'Nuevo León': (25.5922, -99.9962),
        'Coahuila': (27.0587, -101.7068),
        'Oaxaca': (17.0732, -96.7266),
        'Tamaulipas': (23.7369, -99.1411),
        'Sonora': (29.2972, -110.3309),
        'Veracruz': (19.1738, -96.1342),
        'Michoacán': (19.1538, -101.8831),
        'Colima': (19.1223, -103.8840),
        'Yucatán': (20.7099, -89.0943),
        'Zacatecas': (22.7709, -102.5832),
        'Guerrero': (17.4392, -99.5451),
        'Baja California': (30.8406, -115.2838),
        'Jalisco': (20.6597, -103.3496),
        'Morelos': (18.6813, -99.1013),
        'Tabasco': (17.8409, -92.6189),
        'Aguascalientes': (21.8853, -102.2916),
        'Nayarit': (21.7514, -104.8455),
        'Hidalgo': (20.0911, -98.7624),
        'Baja California Sur': (26.0444, -111.6661),
        'Sinaloa': (25.1721, -107.4795),
        'Campeche': (19.8301, -90.5349),
        'Chihuahua': (28.6329, -106.0691),
        'Durango': (24.5593, -104.6580),
        'nan': None,
        'Unknown': None
    }

    # Calculate the counts of completed underwriting by state and gender
    state_gender_counts = df_uos_completed.groupby(['Assigned_State', 'Gender']).size().reset_index(name='count')

    # Add coordinate information to the counts DataFrame
    # Modified lambda to handle None return from get()
    state_gender_counts['Lat'] = state_gender_counts['Assigned_State'].map(lambda x: state_coordinates.get(x, (None, None))[0] if state_coordinates.get(x) is not None else None)
    state_gender_counts['Lon'] = state_gender_counts['Assigned_State'].map(lambda x: state_coordinates.get(x, (None, None))[1] if state_coordinates.get(x) is not None else None)

    # Drop rows with missing coordinates
    state_gender_counts = state_gender_counts.dropna(subset=['Lat', 'Lon'])

    # Create the bubble map
    fig = px.scatter_mapbox(state_gender_counts,
                            lat="Lat",
                            lon="Lon",
                            size="count",
                            color="Gender",
                            hover_name="Assigned_State",
                            hover_data={"count": True, "Lat": False, "Lon": False},
                            size_max=50,
                            zoom=4,
                            height=600,
                            title="Completed Underwriting by State and Gender")

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()

    # Merge Assigned_State from df_dropped into df_loans
    df_loans = pd.merge(
        df_loans,
        df_dropped[['External account H', 'Assigned_State']],
        left_on='Ally ID',
        right_on='External account H',
        how='left'
    )

    # Calculate the average loan amount per assigned state
    average_loan_amount_by_state = df_loans.groupby('Assigned_State')['LoanAmount'].mean().reset_index()
    average_loan_amount_by_state = average_loan_amount_by_state.dropna(subset=['Assigned_State']) # Drop rows where Assigned_State is NaN

    # Create a bar chart using Plotly Express
    fig = px.bar(average_loan_amount_by_state,
                 x='Assigned_State',
                 y='LoanAmount',
                 title='Average Loan Amount per Assigned State',
                 labels={'Assigned_State': 'Assigned State', 'LoanAmount': 'Average Loan Amount'},
                 color='LoanAmount',  # Color bars based on the loan amount
                 color_continuous_scale=px.colors.sequential.Viridis) # Choose a color scale

    fig.update_layout(xaxis_tickangle=-45) # Rotate x-axis labels for better readability
    fig.show()

    # Create a bar chart using Altair
    chart = alt.Chart(average_loan_amount_by_state).mark_bar().encode(
        x=alt.X('Assigned_State:N', title='Assigned State'),
        y=alt.Y('LoanAmount:Q', title='Average Loan Amount'),
        color=alt.Color('LoanAmount:Q', scale=alt.Scale(range='heatmap')), # Color bars based on the loan amount using a heatmap scale
        tooltip=['Assigned_State', 'LoanAmount']
    ).properties(
        title='Average Loan Amount per Assigned State'
    ).interactive()

    chart.display()

    """- Total Loan Amount per State (Horizontal bar graph)"""

    # Calculate the total loan amount per assigned state
    total_loan_amount_by_state = df_loans.groupby('Assigned_State')['LoanAmount'].sum().reset_index()
    total_loan_amount_by_state = total_loan_amount_by_state.dropna(subset=['Assigned_State']) # Drop rows where Assigned_State is NaN
    total_loan_amount_by_state = total_loan_amount_by_state.sort_values('LoanAmount', ascending=False) # Sort for better visualization

    # Create a horizontal bar chart using Plotly Express
    fig = px.bar(total_loan_amount_by_state,
                 x='LoanAmount',
                 y='Assigned_State',
                 orientation='h', # Make it horizontal
                 title='Total Loan Amount per State',
                 labels={'Assigned_State': 'Assigned State', 'LoanAmount': 'Total Loan Amount'},
                 color='LoanAmount',  # Color bars based on the total loan amount
                 color_continuous_scale=px.colors.sequential.Viridis) # Choose a color scale

    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Order the bars by total loan amount
    fig.show()

    # Create a horizontal bar chart using Altair
    chart = alt.Chart(total_loan_amount_by_state).mark_bar().encode(
        y=alt.Y('Assigned_State:N', title='Assigned State', sort='x'), # Make it horizontal and sort by x
        x=alt.X('LoanAmount:Q', title='Total Loan Amount'),
        color=alt.Color('LoanAmount:Q', scale=alt.Scale(range='heatmap')), # Color bars based on the total loan amount using a heatmap scale
        tooltip=['Assigned_State', 'LoanAmount']
    ).properties(
        title='Total Loan Amount per State'
    ).interactive()

    chart.display()

    """**Enrollement**

    Underwriting Overall Status (Aprobado VS Rechazado)
    """

    def plot_underwriting_status():
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_dropped, x="Underwriting overall status", palette="coolwarm")
        plt.title("UOS Aprobado vs. Rechazado")
        plt.xlabel("Underwriting Status")
        plt.ylabel("Número de Usuarios")
        plt.show()

    plot_underwriting_status()

    # Count the occurrences of each status in the 'Underwriting overall status' column
    status_counts = df_dropped['Underwriting overall status'].value_counts()

    # Print the counts for each status
    print("Status Counts:")
    print(status_counts)

    # Alternatively, you can access individual counts like this:
    failed_count = status_counts.get("FAILED", 0)  # Use .get() to handle cases where the status is not present
    completed_count = status_counts.get("COMPLETED", 0)
    in_progress_count = status_counts.get("IN PROGRESS", 0)

    print("\nIndividual Status Counts:")
    print(f"FAILED: {failed_count}")
    print(f"COMPLETED: {completed_count}")

    """Complete Underwriting Overall by Region"""

    region_coordinates = {
        'Centro': (19.4326, -99.1332),            # Ciudad de México
        'Centro-Oriental': (20.0911, -98.7624),   # Hidalgo/Puebla
        'Sureste': (17.0732, -96.7266),           # Oaxaca/Chiapas
        'Centro-Norte': (22.1565, -100.9855),     # San Luis Potosí/Zacatecas
        'Centro-Occidente': (20.6597, -103.3496), # Jalisco/Guanajuato
        'Norte': (27.0587, -101.7068),            # Coahuila
        'Sur': (16.7569, -93.1292),               # Chiapas/Guerrero
        'Noroeste': (29.2972, -110.3309),         # Sonora/Baja California
        'Este': (19.1738, -96.1342),              # Veracruz/Tabasco
        'Occidente': (21.0190, -101.2574),        # Guanajuato/Nayarit
        'Centro-Sur': (18.6813, -99.1013),        # Morelos/Estado de México
        'nan': None,
        'Unknown': None
    }

    # Calculate the counts of completed underwriting by region and gender
    region_gender_counts = df_uos_completed.groupby(['Region', 'Gender']).size().reset_index(name='count')

    # Add coordinate information to the counts DataFrame
    region_gender_counts['Lat'] = region_gender_counts['Region'].map(lambda x: region_coordinates.get(x, (None, None))[0] if region_coordinates.get(x) is not None else None)
    region_gender_counts['Lon'] = region_gender_counts['Region'].map(lambda x: region_coordinates.get(x, (None, None))[1] if region_coordinates.get(x) is not None else None)

    # Drop rows with missing coordinates
    region_gender_counts = region_gender_counts.dropna(subset=['Lat', 'Lon'])

    # Create the bubble map
    fig = px.scatter_mapbox(region_gender_counts,
                            lat="Lat",
                            lon="Lon",
                            size="count",
                            color="Gender",
                            hover_name="Region",
                            hover_data={"count": True, "Lat": False, "Lon": False},
                            size_max=50,
                            zoom=4,
                            height=600,
                            title="Completed Underwriting by Region and Gender")

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()

    # Calculate the counts of completed underwriting by Region and gender
    region_gender_counts = df_uos_completed.groupby(['Region', 'Gender']).size().reset_index(name='count')

    # Create a stacked bar chart for Completed Underwriting by Region and Gender
    fig = px.bar(region_gender_counts,
                 x='Region',
                 y='count',
                 color='Gender',
                 title='Completed Underwriting by Region and Gender',
                 labels={'Region': 'Region', 'count': 'Number of Users', 'Gender': 'Gender'})

    fig.update_layout(xaxis_tickangle=-45) # Rotate x-axis labels for better readability
    fig.show()

    """Complete Counts for Each Status Column"""

    completed_counts = {}
    for col in df_dropped.columns:
        if not pd.api.types.is_numeric_dtype(df_dropped[col]):
            completed_counts[col] = df_dropped[df_dropped[col] == "COMPLETED"][col].count()

    completed_df = pd.DataFrame.from_dict(completed_counts, orient='index', columns=['COMPLETED Count'])
    completed_df_sorted = completed_df.sort_values(by='COMPLETED Count', ascending=False)
    completed_df_sorted

    # Filter out columns with 0 completed counts
    completed_df_filtered = completed_df_sorted[completed_df_sorted['COMPLETED Count'] > 0].reset_index()
    completed_df_filtered.columns = ['Status Column', 'COMPLETED Count']

    # Create a Plotly Express horizontal bar chart showing progression (without numbers)
    fig = px.bar(completed_df_filtered,
                 x='COMPLETED Count',
                 y='Status Column',
                 orientation='h',
                 title='Completed Counts Progression for Each Status Column',
                 labels={'Status Column': 'Status Column'}, # Only show the label for the status column
                 color='COMPLETED Count', # Color indicates the magnitude
                 color_continuous_scale=px.colors.sequential.Viridis)

    # Customize layout to remove numerical labels on the x-axis and add a vertical line
    fig.update_layout(
        xaxis=dict(
            title='Progression (Relative Magnitude)',
            showticklabels=False # Hide tick labels on the x-axis
        ),
        yaxis={'categoryorder':'total ascending'}
    )

    fig.show()

    """# CLEANED

    Data Selection and Exploration
    """

    # Imports
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
    import seaborn as sns

    # Display unique values per column
    for col in df_campaigns.columns:
        print(f"Column: {col}")
        print(df_campaigns[col].unique())
        print("-" * 20)

    # Drop columns with only NaN
    for col in df_campaigns.columns:
        if df_campaigns[col].dropna().nunique() == 0 and df_campaigns[col].isnull().all():
            print(f"Dropping column: {col} because it contains only NaN values.")
            df_campaigns = df_campaigns.drop(columns=[col])

    print("\nDataFrame after dropping columns with only NaN values:")
    print(df_campaigns.head())

    # Replace text-based missing values
    df_campaigns.replace('N A', pd.NA, inplace=True)

    # Standardize and parse date columns
    date_columns = [
        'fecha_entrada', 'fecha_underwriting', 'fecha_sales_info', 'fecha_kyc_done',
        'fecha_referencias', 'fecha_credito', 'fecha_prestamo'
    ]

    # Check for formatting issues
    for col in date_columns:
        if col in df_campaigns.columns:
            print(f"\nInvalid entries in '{col}':")
            invalid_mask = ~df_campaigns[col].astype(str).str.fullmatch(r"\d{8}")
            print(df_campaigns.loc[invalid_mask, col].unique())

    # Convert to datetime
    for col in date_columns:
        if col in df_campaigns.columns:
            df_campaigns[col] = pd.to_datetime(df_campaigns[col].astype('Int64'), format='%Y%m%d', errors='coerce')

    # Check range for 'fecha_entrada'
    if 'fecha_entrada' in df_campaigns.columns:
        print(f"Min fecha_entrada: {df_campaigns['fecha_entrada'].min()}")
        print(f"Max fecha_entrada: {df_campaigns['fecha_entrada'].max()}")

    # Show shape and missing values
    print(df_campaigns.shape)
    print(df_campaigns.isna().sum())

    # Copy to preserve original
    df_campaigns = df_campaigns.copy()

    # Show unique underwriting types if exists
    if 'underwriting_type' in df_campaigns.columns:
        print(df_campaigns['underwriting_type'].unique())

    # Filter out CONCORD underwriting types if available
    if 'underwriting_type' in df_campaigns.columns:
        df_no_concord = df_campaigns[df_campaigns['underwriting_type'] != 'CONCORD'].copy()
        print(df_no_concord.head(), df_no_concord.shape)
    else:
        df_no_concord = df_campaigns.copy()

    # Normalize loan and credit fields
    for col in ['tiene_credito', 'tiene_prestamo']:
        if col in df_no_concord.columns:
            df_no_concord[col] = pd.to_numeric(df_no_concord[col], errors='coerce').fillna(0).astype(int)

    # Fix logical inconsistencies
    if 'tiene_credito' in df_no_concord.columns and 'tiene_prestamo' in df_no_concord.columns:
        before_correction = df_no_concord[
            (df_no_concord["tiene_credito"] == 0) & (df_no_concord["tiene_prestamo"] == 1)
        ].shape[0]

        df_no_concord.loc[df_no_concord["tiene_prestamo"] == 1, "tiene_credito"] = 1

        after_correction = df_no_concord[
            (df_no_concord["tiene_credito"] == 0) & (df_no_concord["tiene_prestamo"] == 1)
        ].shape[0]

        print("Users with loan but no credit:")
        print("Before correction:", before_correction)
        print("After correction:", after_correction)

        print("Combinations of tiene_credito and tiene_prestamo after correction:")
        print(df_no_concord.groupby(['tiene_credito', 'tiene_prestamo']).size().reset_index(name='count'))

    """DT and Concord"""

    df_no_concord["tiene_credito"] = pd.to_numeric(df_campaigns["tiene_credito"], errors='coerce').fillna(0).astype(int)
    df_no_concord["tiene_prestamo"] = pd.to_numeric(df_campaigns["tiene_prestamo"], errors='coerce').fillna(0).astype(int)

    before_correction = df_no_concord[(df_no_concord["tiene_credito"] == 0) & (df_no_concord["tiene_prestamo"] == 1)].shape[0]

    df_no_concord.loc[df_no_concord["tiene_prestamo"] == 1, "tiene_credito"] = 1

    after_correction = df_no_concord[(df_no_concord["tiene_credito"] == 0) & (df_no_concord["tiene_prestamo"] == 1)].shape[0]

    print("Users with loan but no credit:")
    print("Before correction:", before_correction)
    print("After correction:", after_correction)

    print("Combinations of tiene_credito and tiene_prestamo after correction:")
    print(df_no_concord.groupby(['tiene_credito', 'tiene_prestamo']).size().reset_index(name='count'))

    """**Channels**

    Top 10 Marketing Channels (65% User Count, 35% Conversion Rate)
    """

    # User count per marketing channel
    channel_counts = df_no_concord['marketing_channel'].value_counts().reset_index()
    channel_counts.columns = ['marketing_channel', 'user_count']

    # Loan conversion rate per channel
    loanconv_by_channel = df_no_concord.groupby('marketing_channel')['tiene_prestamo'].mean().reset_index()
    loanconv_by_channel.columns = ['marketing_channel', 'conversion_rate']

    # Merge user count and conversion rate
    c_and_c_loan = pd.merge(channel_counts, loanconv_by_channel, on='marketing_channel')

    # Normalize the metrics
    c_and_c_loan['user_count_norm'] = (
        (c_and_c_loan['user_count'] - c_and_c_loan['user_count'].min()) /
        (c_and_c_loan['user_count'].max() - c_and_c_loan['user_count'].min())
    )
    c_and_c_loan['conversion_rate_norm'] = (
        (c_and_c_loan['conversion_rate'] - c_and_c_loan['conversion_rate'].min()) /
        (c_and_c_loan['conversion_rate'].max() - c_and_c_loan['conversion_rate'].min())
    )

    # Weights
    user_weight = 0.60
    conversion_weight = 0.40

    # Combined score
    c_and_c_loan['combined_score'] = (
        user_weight * c_and_c_loan['user_count_norm'] +
        conversion_weight * c_and_c_loan['conversion_rate_norm']
    )

    # Top 10 channels
    top_10_channels_loan = c_and_c_loan.sort_values(by='combined_score', ascending=False).head(10)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='combined_score', y='marketing_channel', data=top_10_channels_loan, palette='viridis')
    plt.title('Top 10 Marketing Channels (60% User Count, 40% Loan Conversion)')
    plt.xlabel('Combined Score (Weighted)')
    plt.ylabel('Marketing Channel')
    plt.tight_layout()
    plt.show()

    """Top 10 Marketing Sub Channels Loans (65% User Count, 35% Conversion Rate)"""

    # User count per marketing subchannel
    schannel_counts = df_no_concord['marketing_subchannel'].value_counts().reset_index()
    schannel_counts.columns = ['marketing_subchannel', 'user_count']

    # Loan conversion rate per subchannel
    loanconv_by_schannel = df_no_concord.groupby('marketing_subchannel')['tiene_prestamo'].mean().reset_index()
    loanconv_by_schannel.columns = ['marketing_subchannel', 'conversion_rate']

    # Merge user count and conversion rate
    sc_and_c_loan = pd.merge(schannel_counts, loanconv_by_schannel, on='marketing_subchannel')

    # Normalize the metrics
    sc_and_c_loan['user_count_norm'] = (
        (sc_and_c_loan['user_count'] - sc_and_c_loan['user_count'].min()) /
        (sc_and_c_loan['user_count'].max() - sc_and_c_loan['user_count'].min())
    )
    sc_and_c_loan['conversion_rate_norm'] = (
        (sc_and_c_loan['conversion_rate'] - sc_and_c_loan['conversion_rate'].min()) /
        (sc_and_c_loan['conversion_rate'].max() - sc_and_c_loan['conversion_rate'].min())
    )

    # Weights
    user_weight = 0.50
    conversion_weight = 0.50

    # Combined score
    sc_and_c_loan['combined_score'] = (
        user_weight * sc_and_c_loan['user_count_norm'] +
        conversion_weight * sc_and_c_loan['conversion_rate_norm']
    )

    # Top 10 subchannels
    top_10_schannels_loan = sc_and_c_loan.sort_values(by='combined_score', ascending=False).head(10)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='combined_score', y='marketing_subchannel', data=top_10_schannels_loan, palette='viridis')
    plt.title('Top 10 Marketing Sub-Channels (50% User Count, 50% Loan Conversion)')
    plt.xlabel('Combined Score (Weighted)')
    plt.ylabel('Marketing Sub-Channel')
    plt.tight_layout()
    plt.show()

    """Are channels and mediums consistent?"""

    df_no_concord['campaign_category'] = df_campaigns.apply(lambda row: row['first_campaign'] if pd.notna(row['first_campaign']) else row['session_manual_campaign_name'] if pd.notna(row['session_manual_campaign_name']) else None, axis=1)

    print(df_no_concord[['first_campaign', 'session_manual_campaign_name', 'campaign_category']].head())

    channel_medium_columns = [col for col in df_no_concord.columns if any(kw in col.lower() for kw in ['source', 'medium', 'channel'])]

    df_cleaned = df_no_concord.copy()

    def clean_text(val):
        if pd.isna(val):
            return "Unknown"
        val = str(val).strip().lower()
        if val in ["(not set)", "(none)", "none", "nan", "null", ""]:
            return "Unknown"
        return val

    channel_medium_columns = [
        'first_source', 'first_medium',
        'session_manual_source', 'session_manual_medium',
        'firebase_source', 'firebase_medium',
        'underwriting_source', 'underwriting_medium',
        'salesinfo_source', 'salesinfo_medium',
        'kyc_start_source', 'kyc_start_medium',
        'kyc_done_source', 'kyc_done_medium',
        'ref_source', 'ref_medium',
        'credito_source', 'credito_medium',
        'prestamo_source', 'prestamo_medium',
        'marketing_channel', 'marketing_subchannel'
    ]

    for col in channel_medium_columns:
        df_cleaned[col] = df_cleaned[col].apply(clean_text)

    print("Cleaned Channel and Medium Fields:")
    print(df_cleaned[channel_medium_columns].head(30))

    df_model_ready = df_cleaned.copy()

    df_model_ready["acquisition_path"] = (
        df_model_ready['marketing_channel'].astype(str).str.lower().str.strip() + " | " +
        df_model_ready["marketing_subchannel"].astype(str).str.lower().str.strip() + " | "
        )

    encode_cols = ['first_source', 'first_medium', 'marketing_channel', 'marketing_subchannel', 'campaign_category']

    df_model_ready[encode_cols] = df_model_ready[encode_cols].fillna("unknown").astype(str).apply(lambda x: x.str.lower().str.strip())
    df_encoded_final = pd.get_dummies(df_model_ready, columns=encode_cols, drop_first=True)

    print("Final Model Dataset (Encoded):")
    print(df_encoded_final.head(30))

    all_funnel_stages = (
        df_no_concord["funnel"]
        .dropna()
        .str.lower()
        .str.split(",")
        .explode()
        .str.strip()
        .dropna()
        .unique()
    )

    all_funnel_stages = sorted(all_funnel_stages)
    all_funnel_stages

    detailed_funnel_order = [
        "underwriting_started",
        "sales_information_completed",
        "kyc_started",
        "kyc_completed",
        "references_completed",
        "credit_line_granted",
        "loan_disbursed"
    ]

    def compute_detailed_funnel_depth(funnel_str):
        if pd.isna(funnel_str):
            return 0
        stages = str(funnel_str).lower().split(",")
        for i in reversed(range(len(detailed_funnel_order))):
            if detailed_funnel_order[i] in stages:
                return i + 1
        return 0

    df_no_concord["funnel_depth_detailed"] = df_no_concord["funnel"].apply(compute_detailed_funnel_depth)

    depth_labels_detailed = {
        i + 1: stage for i, stage in enumerate(detailed_funnel_order)
    }
    depth_labels_detailed[0] = "no_funnel_data"
    df_no_concord["funnel_stage_detailed"] = df_no_concord["funnel_depth_detailed"].map(depth_labels_detailed)

    df_no_concord[["funnel", "funnel_depth_detailed", "funnel_stage_detailed"]].head(30)

    df_time = df_model_ready.copy()

    df_time["dias_a_credito"] = pd.to_numeric(df_time["dias_a_credito"], errors='coerce')
    df_time["dias_a_prestamo"] = pd.to_numeric(df_time["dias_a_prestamo"], errors='coerce')

    df_time.loc[df_time["dias_a_credito"] < 0, "dias_a_credito"] = pd.NA
    df_time.loc[df_time["dias_a_prestamo"] < 0, "dias_a_prestamo"] = pd.NA

    df_time["bin_credito"] = pd.cut(df_time["dias_a_credito"],
                                    bins=[-1, 0, 3, 7, 14, 30, 60, 120, float("inf")],
                                    labels=["0", "1-3", "4-7", "8-14", "15-30", "31-60", "61-120", "120+"],
                                    include_lowest=True)

    df_time["bin_prestamo"] = pd.cut(df_time["dias_a_prestamo"],
                                     bins=[-1, 0, 3, 7, 14, 30, 60, 120, float("inf")],
                                     labels=["0", "1-3", "4-7", "8-14", "15-30", "31-60", "61-120", "120+"],
                                     include_lowest=True)

    print("Time to Conversion (Binned):")
    print(df_time[["dias_a_credito", "dias_a_prestamo", "bin_credito", "bin_prestamo"]].head(30))

    summary_stats = df_time[["dias_a_credito", "dias_a_prestamo"]].describe()

    most_common_credit_bin = df_time["bin_credito"].mode().iloc[0] if not df_time["bin_credito"].mode().empty else None
    most_common_loan_bin = df_time["bin_prestamo"].mode().iloc[0] if not df_time["bin_prestamo"].mode().empty else None

    (summary_stats, most_common_credit_bin, most_common_loan_bin)

    columns_to_drop = ["collected_campaign", "loan_amount", "firebase_campaign_name"]
    df_encoded_final = df_encoded_final.drop(columns=[col for col in columns_to_drop if col in df_encoded_final.columns], errors="ignore")

    columns_to_flag = ["fecha_kyc_start", "credito_campaign"]
    for col in columns_to_flag:
        if col in df_encoded_final.columns:
            df_encoded_final[f"{col}_recorded"] = df_encoded_final[col].notnull().astype(int)

    print("Cleaned Final Model Dataset:")
    print(df_encoded_final.head(30))

    df_encoded_final.shape

    df_paths_credito = df_encoded_final.copy()

    path_summary_credito = (
        df_paths_credito
        .groupby("acquisition_path")
        .agg(
            users=("user_id", "count"),
            credits=("tiene_credito", "sum")
        )
        .reset_index()
    )

    path_summary_credito["credit_conversion_rate"] = (path_summary_credito["credits"] / path_summary_credito["users"] * 100).round(2)
    path_summary_credito_users = path_summary_credito.sort_values(by="users", ascending=False).reset_index(drop=True)

    print("\nAcquisition Path Performance by 'tiene_credito' (Sorted by Users):")
    print(path_summary_credito_users.head(50))

    df_paths_credito = df_encoded_final.copy()

    path_summary_credito = (
        df_paths_credito
        .groupby("acquisition_path")
        .agg(
            users=("user_id", "count"),
            credits=("tiene_credito", "sum")
        )
        .reset_index()
    )

    path_summary_credito["credit_conversion_rate"] = (path_summary_credito["credits"] / path_summary_credito["users"] * 100).round(2)
    path_summary_credito_users = path_summary_credito.sort_values(by="users", ascending=False).reset_index(drop=True)

    print("\nAcquisition Path Performance by 'tiene_credito' (Sorted by Users):")
    print(path_summary_credito_users.head(50))

    # User conversion sorted by loan
    df_paths_prestamo = df_encoded_final.copy()

    path_summary_prestamo = (
        df_paths_prestamo
        .groupby("acquisition_path")
        .agg(
            users=("user_id", "count"),
            loans=("tiene_prestamo", "sum")
        )
        .reset_index()
    )

    path_summary_prestamo["loan_conversion_rate"] = (path_summary_prestamo["loans"] / path_summary_prestamo["users"] * 100).round(2)
    path_summary_prestamo_users = path_summary_prestamo.sort_values(by="users", ascending=False).reset_index(drop=True)

    print("\nAcquisition Path Performance by 'tiene_prestamo' (Sorted by Users):")
    print(path_summary_prestamo_users.head(50))

    top_paths_user_volume = path_summary_prestamo_users.head(20)

    df_paths_prestamo = df_encoded_final.copy()

    path_summary_prestamo = (
        df_paths_prestamo
        .groupby("acquisition_path")
        .agg(
            users=("user_id", "count"),
            loans=("tiene_prestamo", "sum")
        )
        .reset_index()
    )

    path_summary_prestamo["loan_conversion_rate"] = (path_summary_prestamo["loans"] / path_summary_prestamo["users"] * 100).round(2)
    path_summary_prestamo_users = path_summary_prestamo.sort_values(by="users", ascending=False).reset_index(drop=True)

    print("\nAcquisition Path Performance by 'tiene_prestamo' (Sorted by Users):")
    print(path_summary_prestamo_users.head(50))

    path_summary_prestamo_rate = path_summary_prestamo[path_summary_prestamo['users'] > 500].sort_values(by="loan_conversion_rate", ascending=False).reset_index(drop=True)
    print("\nAcquisition Path Performance by 'tiene_prestamo' (Sorted by Loan Conversion Rate - Min 500 Users):")
    print(path_summary_prestamo_rate.head(50))

    top_paths = path_summary_prestamo_rate.sort_values(by=["loan_conversion_rate", "users"], ascending=[False, False]).head(10).copy()

    top_paths.columns = ["Acquisition Path", "User Count", "Conversions", "Conversion Rate (%)"]

    print("Top Performing Acquisition Paths (For Loan Conversion):")
    print(top_paths)

    """Top Acquisition Paths by Loan Conversion Rate ("tiene_prestamo")"""

    top_paths_loan_conversion = path_summary_prestamo_rate.head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='loan_conversion_rate', y='acquisition_path', data=top_paths_loan_conversion, palette='viridis')
    plt.title('Top Acquisition Paths by Loan Conversion Rate ("tiene_prestamo")')
    plt.xlabel('Loan Conversion Rate (%)')
    plt.ylabel('Acquisition Path')
    plt.tight_layout()
    plt.show()

    """Time to Loan  (TABLE)

    Time to Credit (TABLE)

    **ML**

    Machine Learning Random Forest (MODELO y VARIABLES UTILIZADAS, METRICS & CONFUSION MATRIX)
    """

    # Ensure fecha_entrada is datetime
    df_no_concord['fecha_entrada'] = pd.to_datetime(df_no_concord['fecha_entrada'], errors='coerce')

    # Extract month and weekday from fecha_entrada
    df_no_concord['mes_entrada'] = df_no_concord['fecha_entrada'].dt.month
    df_no_concord['dia_semana_entrada'] = df_no_concord['fecha_entrada'].dt.dayofweek

    X_df = df_no_concord[['first_source', 'first_medium', 'first_campaign', 'mes_entrada', 'dia_semana_entrada']].copy()
    y_df = df_no_concord['tiene_prestamo']

    for col in ['first_source', 'first_medium', 'first_campaign']:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, stratify=y_df, test_size=0.3, random_state=42)

    rf_clean = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_clean.fit(X_train, y_train)

    y_probs = rf_clean.predict_proba(X_df)[:, 1]
    optimal_threshold = 0.18
    df_no_concord['predicted_converters'] = (y_probs >= optimal_threshold).astype(int)
    predicted_converters = df_no_concord[df_no_concord['predicted_converters'] == 1].copy()

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure 'fecha_entrada' is datetime
    df_campaigns['fecha_entrada'] = pd.to_datetime(df_campaigns['fecha_entrada'], errors='coerce')

    # Feature engineering: extract month and weekday
    df_campaigns['mes_entrada'] = df_campaigns['fecha_entrada'].dt.month
    df_campaigns['dia_semana_entrada'] = df_campaigns['fecha_entrada'].dt.weekday

    # Drop rows with missing values needed for model
    df_campaigns = df_campaigns.dropna(subset=['tiene_prestamo', 'first_source', 'first_medium', 'first_campaign'])

    # Prepare features and target variable
    X_df = df_campaigns[[
        'first_source', 'first_medium', 'first_campaign',
        'mes_entrada', 'dia_semana_entrada'
    ]].copy()
    y_df = df_campaigns['tiene_prestamo']

    # Encode categorical features
    for col in ['first_source', 'first_medium', 'first_campaign']:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, stratify=y_df, test_size=0.3, random_state=42
    )

    # Train the Random Forest
    rf_clean = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_clean.fit(X_train, y_train)

    # ROC and Precision-Recall curves
    y_probs = rf_clean.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)

    # Plot ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Precision-Recall curve
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.show()

    # Apply threshold to full dataset
    optimal_threshold = 0.60
    y_probs_df = rf_clean.predict_proba(X_df)[:, 1]
    y_preds_df = (y_probs_df >= optimal_threshold).astype(int)

    # Evaluate on the full dataset
    df_report = classification_report(y_df, y_preds_df, output_dict=True)
    df_conf_matrix = confusion_matrix(y_df, y_preds_df)
    df_report_df = pd.DataFrame(df_report).transpose()

    # Final output
    df_report_df, df_conf_matrix

    """Top Marketing Channels by Predicted Loan Conversions

    Top {top_n} Marketing Subchannels by Predicted Loan Conversions (Stacked Bar Chart with Top Marketing Channels)
    """

    # --- Plotly: Top Marketing Channels ---
    channel_summary_plotly = predicted_converters.groupby('marketing_channel') \
        .size().reset_index(name='predicted_converters') \
        .sort_values(by='predicted_converters', ascending=True) # Sort ascending for Plotly horizontal bars

    fig_channels = px.bar(channel_summary_plotly,
                          y='marketing_channel',
                          x='predicted_converters',
                          orientation='h',
                          title='Top Marketing Channels by Predicted Loan Conversions',
                          labels={'marketing_channel': 'Marketing Channel', 'predicted_converters': 'Predicted Converters'})

    fig_channels.update_layout(yaxis={'categoryorder': 'total ascending'}) # Order bars by value
    fig_channels.show()

    # --- Plotly: Top Marketing Subchannels ---
    subchannel_summary_plotly = predicted_converters.groupby('marketing_subchannel') \
        .size().reset_index(name='predicted_converters') \
        .sort_values(by='predicted_converters', ascending=True) # Sort ascending for Plotly horizontal bars

    top_n_plotly = 10 # show top 10 for clarity
    subchannel_summary_plotly = subchannel_summary_plotly.tail(top_n_plotly) # Get the top N after sorting ascending

    fig_subchannels = px.bar(subchannel_summary_plotly,
                             y='marketing_subchannel',
                             x='predicted_converters',
                             orientation='h',
                             title=f'Top {top_n_plotly} Marketing Subchannels by Predicted Loan Conversions',
                             labels={'marketing_subchannel': 'Marketing Subchannel', 'predicted_converters': 'Predicted Converters'})

    fig_subchannels.update_layout(yaxis={'categoryorder': 'total ascending'}) # Order bars by value
    fig_subchannels.show()

    """Top First Sources by Predicted Loan Conversions (Packed Bubble Chart)

    Top First Mediums by Predicted Loan Conversions' (Packed Bubble Chart)

    Top {top_n_campaigns} First Campaigns by Predicted Loan Conversions (Pareto Chart or Colored Bar Chart by Tier)
    """

    # --- Top First Source ---
    source_summary = predicted_converters.groupby('first_source') \
        .size().reset_index(name='predicted_converters') \
        .sort_values(by='predicted_converters', ascending=False)

    # --- Top First Medium ---
    medium_summary = predicted_converters.groupby('first_medium') \
        .size().reset_index(name='predicted_converters') \
        .sort_values(by='predicted_converters', ascending=False)

    # --- Top First Campaign ---
    campaign_summary = predicted_converters.groupby('first_campaign') \
        .size().reset_index(name='predicted_converters') \
        .sort_values(by='predicted_converters', ascending=False)

    # --- Plot: First Source ---
    plt.figure(figsize=(10, 5))
    plt.barh(source_summary['first_source'], source_summary['predicted_converters'])
    plt.gca().invert_yaxis()
    plt.xlabel('Predicted Converters')
    plt.title('Top First Sources by Predicted Loan Conversions')
    plt.tight_layout()
    plt.show()

    # --- Plot: First Medium ---
    plt.figure(figsize=(10, 5))
    plt.barh(medium_summary['first_medium'], medium_summary['predicted_converters'])
    plt.gca().invert_yaxis()
    plt.xlabel('Predicted Converters')
    plt.title('Top First Mediums by Predicted Loan Conversions')
    plt.tight_layout()
    plt.show()

    # --- Plot: First Campaign (Top N for clarity) ---
    top_n_campaigns = 10
    plt.figure(figsize=(10, 6))
    plt.barh(campaign_summary['first_campaign'].head(top_n_campaigns),
             campaign_summary['predicted_converters'].head(top_n_campaigns))
    plt.gca().invert_yaxis()
    plt.xlabel('Predicted Converters')
    plt.title(f'Top {top_n_campaigns} First Campaigns by Predicted Loan Conversions')
    plt.tight_layout()
    plt.show()

    # Return summary tables if needed
    source_summary.reset_index(drop=True), medium_summary.reset_index(drop=True), campaign_summary.reset_index(drop=True)

    """Predicted Loan Conversions by Day of Week' (Line Graph)"""

    # Group by Day of Week
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_summary = predicted_converters.groupby('dia_semana_entrada').size().reset_index(name='predicted_converters')
    # Ensure all days are present, even if no conversions on that day
    all_days = pd.DataFrame({'dia_semana_entrada': range(7)})
    weekday_summary = pd.merge(all_days, weekday_summary, on='dia_semana_entrada', how='left').fillna(0)
    weekday_summary = weekday_summary.sort_values(by='dia_semana_entrada')

    # Map numeric day of week to labels
    weekday_summary['day_name'] = weekday_summary['dia_semana_entrada'].map({i: day_labels[i] for i in range(7)})

    # Create Plotly Line Graph
    fig_weekday = px.line(weekday_summary,
                          x='day_name',
                          y='predicted_converters',
                          markers=True,  # Add markers for each data point
                          title='Predicted Loan Conversions by Day of Week',
                          labels={'day_name': 'Day of Week', 'predicted_converters': 'Predicted Converters'})

    # Update layout for better readability and to ensure all days are shown in order
    fig_weekday.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=day_labels
        )
    )

    fig_weekday.show()