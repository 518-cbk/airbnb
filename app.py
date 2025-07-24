import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.title("Airbnb Price Prediction Model")
st.write("This model helps Airbnb owners estimate their listing prices.")

# Load the saved components with error handling
@st.cache_resource
def load_models():
    try:
        # Try to load the clean versions first
        model = joblib.load('model_clean.pkl')
        scaler = joblib.load('scaler_clean.pkl')
        label_encoders = joblib.load('label_encoders_clean.pkl')
        
        # Verify they loaded correctly
        st.success("All models loaded successfully!")
        st.write(f"Model type: {type(model)}")
        st.write(f"Scaler type: {type(scaler)}")
        st.write(f"Encoders loaded: {list(label_encoders.keys())}")
        
        return model, scaler, label_encoders
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure you've run the model saving code in your notebook first.")
        st.stop()

# Load models
model, scaler, label_encoders = load_models()

# Create neighborhood-borough mapping
@st.cache_data
def create_neighborhood_mapping():
    """
    Create a mapping of neighborhoods to boroughs.
    In production, this should come from your training data.
    """
    # This is a sample mapping - you should replace this with actual data from your training set
    neighborhood_borough_map = {
        # Manhattan
        'Midtown': 'Manhattan',
        'Harlem': 'Manhattan', 
        'East Harlem': 'Manhattan',
        'Murray Hill': 'Manhattan',
        "Hell's Kitchen": 'Manhattan',
        'Upper West Side': 'Manhattan',
        'Chinatown': 'Manhattan',
        'West Village': 'Manhattan',
        'Chelsea': 'Manhattan',
        'Inwood': 'Manhattan',
        'East Village': 'Manhattan',
        'Lower East Side': 'Manhattan',
        'Kips Bay': 'Manhattan',
        'SoHo': 'Manhattan',
        'Upper East Side': 'Manhattan',
        'Washington Heights': 'Manhattan',
        'Financial District': 'Manhattan',
        'Morningside Heights': 'Manhattan',
        'NoHo': 'Manhattan',
        'Flatiron District': 'Manhattan',
        'Roosevelt Island': 'Manhattan',
        'Greenwich Village': 'Manhattan',
        'Little Italy': 'Manhattan',
        'Two Bridges': 'Manhattan',
        'Nolita': 'Manhattan',
        'Gramercy': 'Manhattan',
        'Theater District': 'Manhattan',
        'Tribeca': 'Manhattan',
        'Battery Park City': 'Manhattan',
        'Civic Center': 'Manhattan',
        'Stuyvesant Town': 'Manhattan',
        'Marble Hill': 'Manhattan',
        
        # Brooklyn
        'Kensington': 'Brooklyn',
        'Clinton Hill': 'Brooklyn',
        'Bedford-Stuyvesant': 'Brooklyn',
        'South Slope': 'Brooklyn',
        'Fort Greene': 'Brooklyn',
        'Crown Heights': 'Brooklyn',
        'Park Slope': 'Brooklyn',
        'Windsor Terrace': 'Brooklyn',
        'Williamsburg': 'Brooklyn',
        'Greenpoint': 'Brooklyn',
        'Bushwick': 'Brooklyn',
        'Flatbush': 'Brooklyn',
        'Prospect-Lefferts Gardens': 'Brooklyn',
        'Brooklyn Heights': 'Brooklyn',
        'Carroll Gardens': 'Brooklyn',
        'Gowanus': 'Brooklyn',
        'Flatlands': 'Brooklyn',
        'Cobble Hill': 'Brooklyn',
        'Boerum Hill': 'Brooklyn',
        'DUMBO': 'Brooklyn',
        'Prospect Heights': 'Brooklyn',
        'East Flatbush': 'Brooklyn',
        'Gravesend': 'Brooklyn',
        'East New York': 'Brooklyn',
        'Sheepshead Bay': 'Brooklyn',
        'Fort Hamilton': 'Brooklyn',
        'Bensonhurst': 'Brooklyn',
        'Sunset Park': 'Brooklyn',
        'Brighton Beach': 'Brooklyn',
        'Cypress Hills': 'Brooklyn',
        'Bay Ridge': 'Brooklyn',
        'Columbia St': 'Brooklyn',
        'Vinegar Hill': 'Brooklyn',
        'Canarsie': 'Brooklyn',
        'Borough Park': 'Brooklyn',
        'Downtown Brooklyn': 'Brooklyn',
        'Red Hook': 'Brooklyn',
        'Dyker Heights': 'Brooklyn',
        'Sea Gate': 'Brooklyn',
        'Navy Yard': 'Brooklyn',
        'Brownsville': 'Brooklyn',
        'Manhattan Beach': 'Brooklyn',
        'Bergen Beach': 'Brooklyn',
        'Coney Island': 'Brooklyn',
        'Bath Beach': 'Brooklyn',
        'Mill Basin': 'Brooklyn',
        'Breezy Point': 'Brooklyn',
        
        # Queens
        'Long Island City': 'Queens',
        'Woodside': 'Queens',
        'Flushing': 'Queens',
        'Sunnyside': 'Queens',
        'Ridgewood': 'Queens',
        'Jamaica': 'Queens',
        'Middle Village': 'Queens',
        'Ditmars Steinway': 'Queens',
        'Astoria': 'Queens',
        'Queens Village': 'Queens',
        'Rockaway Beach': 'Queens',
        'Forest Hills': 'Queens',
        'Woodlawn': 'Queens',
        'Elmhurst': 'Queens',
        'Jackson Heights': 'Queens',
        'St. Albans': 'Queens',
        'Rego Park': 'Queens',
        'Briarwood': 'Queens',
        'Ozone Park': 'Queens',
        'East Elmhurst': 'Queens',
        'Arverne': 'Queens',
        'Cambria Heights': 'Queens',
        'Bayside': 'Queens',
        'College Point': 'Queens',
        'Glendale': 'Queens',
        'Richmond Hill': 'Queens',
        'Bellerose': 'Queens',
        'Maspeth': 'Queens',
        'Woodhaven': 'Queens',
        'Kew Gardens Hills': 'Queens',
        'Whitestone': 'Queens',
        'Bayswater': 'Queens',
        'Fresh Meadows': 'Queens',
        'Springfield Gardens': 'Queens',
        'Howard Beach': 'Queens',
        'Belle Harbor': 'Queens',
        'Jamaica Estates': 'Queens',
        'Far Rockaway': 'Queens',
        'South Ozone Park': 'Queens',
        'Corona': 'Queens',
        'Neponsit': 'Queens',
        'Laurelton': 'Queens',
        'Holliswood': 'Queens',
        'Rosedale': 'Queens',
        'Edgemere': 'Queens',
        'Jamaica Hills': 'Queens',
        'Hollis': 'Queens',
        'Douglaston': 'Queens',
        'Little Neck': 'Queens',
        'Kew Gardens': 'Queens',
        
        # Bronx
        'Highbridge': 'Bronx',
        'Clason Point': 'Bronx',
        'Eastchester': 'Bronx',
        'Kingsbridge': 'Bronx',
        'University Heights': 'Bronx',
        'Allerton': 'Bronx',
        'Concourse Village': 'Bronx',
        'Concourse': 'Bronx',
        'Wakefield': 'Bronx',
        'Spuyten Duyvil': 'Bronx',
        'Mott Haven': 'Bronx',
        'Longwood': 'Bronx',
        'Morris Heights': 'Bronx',
        'Port Morris': 'Bronx',
        'Fieldston': 'Bronx',
        'Mount Eden': 'Bronx',
        'City Island': 'Bronx',
        'Williamsbridge': 'Bronx',
        'Soundview': 'Bronx',
        'Woodrow': 'Bronx',
        'Co-op City': 'Bronx',
        'Parkchester': 'Bronx',
        'North Riverdale': 'Bronx',
        'Bronxdale': 'Bronx',
        'Riverdale': 'Bronx',
        'Norwood': 'Bronx',
        'Claremont Village': 'Bronx',
        'Fordham': 'Bronx',
        'Mount Hope': 'Bronx',
        'Van Nest': 'Bronx',
        'Morris Park': 'Bronx',
        'Tremont': 'Bronx',
        'East Morrisania': 'Bronx',
        'Hunts Point': 'Bronx',
        'Pelham Bay': 'Bronx',
        'Throgs Neck': 'Bronx',
        'West Farms': 'Bronx',
        'Morrisania': 'Bronx',
        'Pelham Gardens': 'Bronx',
        'Belmont': 'Bronx',
        'Baychester': 'Bronx',
        'Melrose': 'Bronx',
        'Schuylerville': 'Bronx',
        'Castle Hill': 'Bronx',
        'Olinville': 'Bronx',
        'Edenwald': 'Bronx',
        'Westchester Square': 'Bronx',
        'Unionport': 'Bronx',
        
        # Staten Island
        'St. George': 'Staten Island',
        'Tompkinsville': 'Staten Island',
        'Emerson Hill': 'Staten Island',
        'Shore Acres': 'Staten Island',
        'Arrochar': 'Staten Island',
        'Clifton': 'Staten Island',
        'Graniteville': 'Staten Island',
        'Stapleton': 'Staten Island',
        'New Springville': 'Staten Island',
        'Tottenville': 'Staten Island',
        'Mariners Harbor': 'Staten Island',
        'Concord': 'Staten Island',
        'Port Richmond': 'Staten Island',
        'Bay Terrace': 'Staten Island',
        'West Brighton': 'Staten Island',
        'Great Kills': 'Staten Island',
        'Dongan Hills': 'Staten Island',
        'Castleton Corners': 'Staten Island',
        'Randall Manor': 'Staten Island',
        'Todt Hill': 'Staten Island',
        'Silver Lake': 'Staten Island',
        'Grymes Hill': 'Staten Island',
        'New Brighton': 'Staten Island',
        'Midland Beach': 'Staten Island',
        'Richmondtown': 'Staten Island',
        'Howland Hook': 'Staten Island',
        'New Dorp Beach': 'Staten Island',
        "Prince's Bay": 'Staten Island',
        'South Beach': 'Staten Island',
        'Oakwood': 'Staten Island',
        'Grant City': 'Staten Island',
        'Westerleigh': 'Staten Island',
        'Bay Terrace, Staten Island': 'Staten Island',
        'Fort Wadsworth': 'Staten Island',
        'Rosebank': 'Staten Island',
        'Arden Heights': 'Staten Island',
        "Bull's Head": 'Staten Island',
        'New Dorp': 'Staten Island',
        'Rossville': 'Staten Island',
        'Willowbrook': 'Staten Island',
        'Lighthouse Hill': 'Staten Island',
        'Huguenot': 'Staten Island'
    }
    
    # Create reverse mapping (borough -> list of neighborhoods)
    borough_neighborhoods = {}
    for neighborhood, borough in neighborhood_borough_map.items():
        if borough not in borough_neighborhoods:
            borough_neighborhoods[borough] = []
        borough_neighborhoods[borough].append(neighborhood)
    
    return neighborhood_borough_map, borough_neighborhoods

# Alternative: Load from your training data (BETTER APPROACH)
@st.cache_data
def load_neighborhood_mapping_from_data():
    """
    Load the actual neighborhood-borough mapping from your training data.
    This is the recommended approach.
    """
    try:
        # Try to load from a saved mapping file
        with open('neighborhood_borough_mapping.json', 'r') as f:
            mapping_data = json.load(f)
            return mapping_data['neighborhood_to_borough'], mapping_data['borough_to_neighborhoods']
    except:
        # Fallback to hardcoded mapping
        return create_neighborhood_mapping()

neighborhood_borough_map, borough_neighborhoods = load_neighborhood_mapping_from_data()

# Get available options from encoders
try:
    all_neighborhoods = list(label_encoders['neighbourhood'].classes_)
    all_boroughs = list(label_encoders['neighbourhood_group'].classes_)
except:
    # Fallback options
    all_neighborhoods = list(neighborhood_borough_map.keys())
    all_boroughs = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']

# User inputs with filtering
st.subheader("Location Selection")

# First, let user select borough
neighbourhood_group = st.selectbox('Borough (Neighbourhood Group)', all_boroughs)

# Filter neighborhoods based on selected borough
if neighbourhood_group in borough_neighborhoods:
    available_neighborhoods = borough_neighborhoods[neighbourhood_group]
else:
    # If no mapping exists, show all neighborhoods
    available_neighborhoods = all_neighborhoods
    st.warning(f"No neighborhood mapping found for {neighbourhood_group}. Showing all neighborhoods.")

selected_neighborhood = st.selectbox(
    'Neighborhood', 
    available_neighborhoods,
    help=f"Neighborhoods available in {neighbourhood_group}"
)

# Show the automatic mapping
if selected_neighborhood in neighborhood_borough_map:
    mapped_borough = neighborhood_borough_map[selected_neighborhood]
    if mapped_borough != neighbourhood_group:
        st.warning(f"Note: {selected_neighborhood} is actually in {mapped_borough}. Auto-correcting borough selection.")
        neighbourhood_group = mapped_borough

# For room_type
try:
    room_type_options = list(label_encoders['room_type'].classes_)
except:
    room_type_options = ['Private room', 'Entire home/apt', 'Shared room']

st.subheader("Property Details")
room_type = st.selectbox('Room Type', room_type_options)

# Numeric inputs
st.subheader("Property Specifications")
col1, col2 = st.columns(2)

with col1:
    longitude = st.slider('Longitude', -74.95, -73.70, value=-74.0)
    latitude = st.slider('Latitude', 40.5, 40.9, value=40.7)
    minimum_nights = st.slider('Minimum Nights', 1, 1250, value=1)
    number_of_reviews = st.slider('Number of Reviews', 0, 629, value=0)

with col2:
    calculated_host_listings_count = st.slider('Host Listings Count', 1, 327, value=1)
    availability_365 = st.slider('Availability (days)', 0, 365, value=365)

# Make prediction
if st.button('🔮 Predict Airbnb Price', type="primary"):
    try:
        # Encode categorical features using the loaded encoders
        neighbourhood_group_encoded = label_encoders['neighbourhood_group'].transform([neighbourhood_group])[0]
        neighborhood_encoded = label_encoders['neighbourhood'].transform([selected_neighborhood])[0]
        room_type_encoded = label_encoders['room_type'].transform([room_type])[0]
        
        st.write("✓ Categorical features encoded successfully")
        
        # Create numerical features array for scaling
        numerical_features = np.array([[minimum_nights, number_of_reviews, calculated_host_listings_count, 
                                       availability_365, longitude, latitude]])
        
        # Scale the numerical features
        scaled_numerical = scaler.transform(numerical_features)
        st.write("✓ Numerical features scaled successfully")
        
        # Create final feature array in the order expected by the model
        features_for_prediction = np.array([[
            neighbourhood_group_encoded,    # neighbourhood_group
            neighborhood_encoded,           # neighbourhood  
            room_type_encoded,             # room_type
            scaled_numerical[0][5],        # latitude (scaled)
            scaled_numerical[0][4],        # longitude (scaled)
            scaled_numerical[0][0],        # minimum_nights (scaled)
            scaled_numerical[0][1],        # number_of_reviews (scaled)
            scaled_numerical[0][2],        # calculated_host_listings_count (scaled)
            scaled_numerical[0][3]         # availability_365 (scaled)
        ]])
        
        # Make prediction
        prediction = model.predict(features_for_prediction)
        predicted_price = prediction[0]
        
        # Display results with better formatting
        st.success(f'🎉 **Predicted Price: ${predicted_price:.2f} per night**')
        
        # Show input summary in an organized way
        with st.expander("📋 Input Summary", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Location:**")
                st.write(f"• Borough: {neighbourhood_group}")
                st.write(f"• Neighborhood: {selected_neighborhood}")
                st.write(f"• Coordinates: ({latitude:.3f}, {longitude:.3f})")
                
                st.write("**Property:**")
                st.write(f"• Room Type: {room_type}")
                st.write(f"• Minimum Nights: {minimum_nights}")
            
            with col2:
                st.write("**Host & Reviews:**")
                st.write(f"• Number of Reviews: {number_of_reviews}")
                st.write(f"• Host Listings Count: {calculated_host_listings_count}")
                
                st.write("**Availability:**")
                st.write(f"• Available Days/Year: {availability_365}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        
        with st.expander("🔍 Debug Information"):
            import traceback
            st.code(traceback.format_exc())

# Add information about the mapping
with st.expander("ℹ️ About Neighborhood-Borough Mapping"):
    st.write("""
    This app automatically filters neighborhoods based on the selected borough to ensure 
    geographic consistency. The mapping is based on the training data used to build the model.
    """)
    
    # Show current mapping for selected borough
    if neighbourhood_group in borough_neighborhoods:
        st.write(f"**Neighborhoods in {neighbourhood_group}:**")
        for neighborhood in sorted(borough_neighborhoods[neighbourhood_group]):
            st.write(f"• {neighborhood}")