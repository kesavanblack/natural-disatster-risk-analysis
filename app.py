import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from streamlit_folium import folium_static
import plotly.express as px

# Load Data
@st.cache
def load_data():
    data = pd.read_csv("processed_disaster_data.csv")  # Processed dataset
    return data

# Preprocess Data
def preprocess_data(data):
    X = data[['latitude', 'longitude', 'magnitude', 'deaths', 'affected', 'damages']]
    y = data['risk']  # Binary: 0 (Low risk), 1 (High risk)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Create Map
def create_map_with_click(data, predictions=None):
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=6)
    for i, row in data.iterrows():
        color = 'red' if predictions is not None and predictions[i] == 1 else 'green'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
    m.add_child(folium.ClickForMarker(popup="Selected Location"))
    return m

# Streamlit App
st.title("Natural Disaster Risk Analysis")
st.markdown("### Predict and visualize disaster risks across regions.")

# Load Data
data = load_data()

# Country Search and Selection
if 'country' in data.columns:  # Ensure country data exists
    country_list = sorted(data['country'].dropna().unique())
    selected_country = st.sidebar.selectbox("Search or Select Country", ["All"] + country_list)
    
    if selected_country != "All":
        data = data[data['country'] == selected_country]

# Display Dataset
st.write("### Dataset Preview", data.head())

# Train-Test Split
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train Model
model = train_model(X_train, y_train)

# Predict
st.write("### Model Accuracy")
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))

# Visualize Map
if st.checkbox("Show Disaster Risk Map"):
    predictions = model.predict(data[['latitude', 'longitude', 'magnitude', 'deaths', 'affected', 'damages']])
    disaster_map = create_map_with_click(data, predictions)
    folium_static(disaster_map)

# Additional Plots
st.markdown("### Data Insights")

# Bar Plot: Total Disasters by Type
# Bar Plot: Total Disasters by Type
if 'type' in data.columns:  # Ensure type column exists
    disaster_count = data['type'].value_counts().reset_index()
    disaster_count.columns = ['Disaster Type', 'Count']
    bar_fig = px.bar(disaster_count, x='Disaster Type', y='Count', title="Total Disasters by Type")
    st.plotly_chart(bar_fig)

# Scatter Plot: Magnitude vs. Damages
if 'magnitude' in data.columns and 'damages' in data.columns and 'risk' in data.columns:
    scatter_fig = px.scatter(
        data,
        x='magnitude',
        y='damages',
        color='risk',
        title="Magnitude vs. Damages (Colored by Risk)",
        labels={'magnitude': 'Magnitude', 'damages': 'Damages ($)', 'risk': 'Risk Level'}
    )
    st.plotly_chart(scatter_fig)

# Line Plot: Deaths Over the Years
# Bar Plot: Total Deaths by Disaster Type
# Pie Chart: Total Deaths by Disaster Type
if 'deaths' in data.columns and 'type' in data.columns:
    deaths_by_type = data.groupby('type')['deaths'].sum().reset_index()
    deaths_by_type.columns = ['Disaster Type', 'Total Deaths']
    deaths_pie_fig = px.pie(
        deaths_by_type,
        names='Disaster Type',
        values='Total Deaths',
        title="Proportion of Total Deaths by Disaster Type",
        labels={'Disaster Type': 'Disaster Type', 'Total Deaths': 'Deaths'},
        hole=0.4  # Optional: For a donut-style chart
    )
    st.plotly_chart(deaths_pie_fig)
else:
    st.write("The dataset does not contain the necessary columns for this plot.")

# User Input
st.sidebar.markdown("### Custom Prediction")
st.sidebar.write("Click on the map to select a location.")
lat = st.sidebar.number_input("Latitude", key="lat", value=0.0)
lon = st.sidebar.number_input("Longitude", key="lon", value=0.0)
mag = st.sidebar.number_input("Magnitude")
deaths = st.sidebar.number_input("Deaths")
affected = st.sidebar.number_input("People Affected")
damages = st.sidebar.number_input("Damages ($)")

if st.sidebar.button("Predict Risk"):
    custom_input = np.array([[lat, lon, mag, deaths, affected, damages]])
    risk = model.predict(custom_input)
    st.sidebar.write("Risk Level: High" if risk[0] == 1 else "Risk Level: Low")
