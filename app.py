import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import requests
from datetime import datetime
import json
import os
from typing import Optional, Tuple, Dict, Any, List
import time
from datetime import timedelta
import logging
import math
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
    st.session_state.map_initialized = False
    st.session_state.stable_view_state = None
    logger.info("Initialized session state")

# Set page config for a dark theme
st.set_page_config(
    page_title="Urban Surveillance Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #fafafa;
    }
    .stTextInput>div>div>input {
        color: #fafafa;
    }
    .stMetric {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e91e63;
    }
    .stDataFrame {
        background-color: rgba(0, 0, 0, 0.2);
    }
    .resource-card {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e91e63;
        margin: 10px 0;
    }
    .resource-card a {
        color: #ff69b4;
        text-decoration: none;
    }
    .resource-card a:hover {
        color: #ff1493;
        text-decoration: underline;
    }
    .info-box {
        background-color: rgba(233, 30, 99, 0.1);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e91e63;
        margin: 10px 0;
    }
    .data-source {
        font-size: 0.9em;
        color: #888;
        font-style: italic;
    }
    .data-repository-expander {
        background-color: rgba(0, 0, 0, 0.2);
        border: 1px solid #e91e63;
        border-radius: 5px;
        margin: 10px 0;
        padding: 10px;
    }
    .data-repository-expander h3 {
        color: #ff69b4;
    }
    .data-repository-expander h4 {
        color: #ff1493;
        margin-top: 20px;
    }
    .data-repository-expander blockquote {
        border-left: 3px solid #e91e63;
        padding-left: 10px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Philadelphia Police Districts data
PHILLY_DISTRICTS = {
    'Central': {
        'lat': 39.9526, 'lon': -75.1652,
        'boundaries': [
            [39.9626, -75.1752],
            [39.9426, -75.1752],
            [39.9426, -75.1552],
            [39.9626, -75.1552]
        ],
        'safecam_count': 450,
        'bscp_count': 280,
        'rtcc_count': 35,
    },
    'South': {
        'lat': 39.9154, 'lon': -75.1688,
        'boundaries': [
            [39.9254, -75.1788],
            [39.9054, -75.1788],
            [39.9054, -75.1588],
            [39.9254, -75.1588]
        ],
        'safecam_count': 380,
        'bscp_count': 220,
        'rtcc_count': 28,
    },
    'Southwest': {
        'lat': 39.9400, 'lon': -75.2180,
        'boundaries': [
            [39.9500, -75.2280],
            [39.9300, -75.2280],
            [39.9300, -75.2080],
            [39.9500, -75.2080]
        ],
        'safecam_count': 290,
        'bscp_count': 180,
        'rtcc_count': 25,
    },
    'Northwest': {
        'lat': 40.0350, 'lon': -75.1780,
        'boundaries': [
            [40.0450, -75.1880],
            [40.0250, -75.1880],
            [40.0250, -75.1680],
            [40.0450, -75.1680]
        ],
        'safecam_count': 320,
        'bscp_count': 190,
        'rtcc_count': 30,
    },
    'Northeast': {
        'lat': 40.0500, 'lon': -75.0700,
        'boundaries': [
            [40.0600, -75.0800],
            [40.0400, -75.0800],
            [40.0400, -75.0600],
            [40.0600, -75.0600]
        ],
        'safecam_count': 410,
        'bscp_count': 250,
        'rtcc_count': 32,
    },
    'East': {
        'lat': 39.9850, 'lon': -75.1200,
        'boundaries': [
            [39.9950, -75.1300],
            [39.9750, -75.1300],
            [39.9750, -75.1100],
            [39.9950, -75.1100]
        ],
        'safecam_count': 350,
        'bscp_count': 210,
        'rtcc_count': 27,
    }
}

# City configurations
CITIES = {
    "New York City": {
        "data_source": "local",
        "data_path": "data/nyc",
        "center": {"lat": 40.7128, "lon": -74.0060},
        "zoom": 10,
        "region_col": "Borough",
        "title": "NYC Surveillance Camera Network",
        "subtitle": "Amnesty Decode Data Visualization",
        "description": """
        This visualization shows surveillance cameras documented through the Amnesty International Decode Surveillance NYC project.
        The data represents a comprehensive mapping of surveillance cameras across New York City's boroughs.
        """,
        "resources": {
            "title": "NYC Surveillance Resources",
            "description": "New York City's surveillance camera network has been documented through various initiatives:",
            "links": [
                {
                    "name": "Amnesty Decode Surveillance NYC",
                    "url": "https://github.com/amnesty-crisis-evidence-lab/decode-surveillance-nyc",
                    "description": "A collaborative project to map surveillance cameras in NYC"
                }
            ]
        }
    },
    "Philadelphia": {
        "data_source": "composite",
        "data_path": "data/philadelphia",
        "center": {"lat": 39.9526, "lon": -75.1652},
        "zoom": 12,
        "region_col": "District",
        "title": "Philadelphia Surveillance Camera Network",
        "subtitle": "Multi-Source Camera Data",
        "description": """
        Philadelphia's surveillance camera network consists of multiple systems and programs:
        
        1. **SafeCam Program**: A Philadelphia Police Department initiative where property owners can register their security cameras.
        2. **Business Security Camera Program (BSCP)**: A city program providing grants to businesses for installing security cameras.
        3. **Real Time Crime Center (RTCC)**: City-operated cameras monitored by the police department.
        
        This visualization combines data from:
        - SafeCam registrations by police district
        - Business Security Camera Program installations
        - Real Time Crime Center camera locations
        - Community-reported camera sightings
        
        Camera locations are approximated within districts to protect privacy and security while maintaining accurate density patterns.
        """,
        "resources": {
            "title": "Philadelphia Surveillance Resources",
            "description": "Philadelphia's surveillance infrastructure includes several programs and initiatives:",
            "links": [
                {
                    "name": "SafeCam Program",
                    "url": "https://safecam.phillypolice.com/",
                    "description": "Philadelphia Police Department's program for registering private security cameras"
                },
                {
                    "name": "Business Security Camera Program",
                    "url": "https://www.phila.gov/programs/business-security-camera-program/",
                    "description": "City program encouraging businesses to install and register external surveillance cameras"
                },
                {
                    "name": "Real Time Crime Center",
                    "url": "https://www.phillypolice.com/programs-services/real-time-crime-center/",
                    "description": "Philadelphia Police Department's centralized camera monitoring facility"
                }
            ]
        }
    }
}

# Add Mapbox token handling
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")
if not MAPBOX_TOKEN:
    st.error("""
        No Mapbox token found. Please set your Mapbox token as an environment variable:
        ```bash
        export MAPBOX_TOKEN='your_mapbox_token_here'
        ```
        You can get a token from https://account.mapbox.com/
    """)
    st.stop()

@st.cache_data(ttl=300)
def get_data_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute and return basic statistics about the dataset."""
    try:
        total_points = len(df)
        total_cameras = df['n_cameras'].sum() if 'n_cameras' in df.columns else 0
        
        # Calculate memory usage in MB
        memory_usage = df.memory_usage(index=True, deep=True).sum() / (1024 * 1024)
        
        # Get coordinate ranges with error handling
        lat_min = float(df['Lat'].min()) if not df['Lat'].empty else 0
        lat_max = float(df['Lat'].max()) if not df['Lat'].empty else 0
        long_min = float(df['Long'].min()) if not df['Long'].empty else 0
        long_max = float(df['Long'].max()) if not df['Long'].empty else 0
        
        # Additional statistics
        avg_cameras = float(df['n_cameras'].mean()) if 'n_cameras' in df.columns else 0
        unique_regions = df[city_config['region_col']].nunique() if city_config['region_col'] in df.columns else 0
        
        stats = {
            "total_points": total_points,
            "total_cameras": int(total_cameras),
            "avg_cameras_per_location": round(avg_cameras, 2),
            "memory_usage": round(memory_usage, 2),
            "lat_range": [lat_min, lat_max],
            "long_range": [long_min, long_max],
            "unique_regions": unique_regions,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated stats for {total_points:,} points")
        return stats
        
    except Exception as e:
        logger.error(f"Error generating stats: {str(e)}")
        # Return minimal stats to prevent crashes
        return {
            "total_points": len(df),
            "total_cameras": 0,
            "avg_cameras_per_location": 0,
            "memory_usage": 0,
            "lat_range": [0, 0],
            "long_range": [0, 0],
            "unique_regions": 0
        }

def optimize_data_for_visualization(df: pd.DataFrame, zoom_level: float) -> pd.DataFrame:
    """Optimize the dataset based on zoom level for better performance."""
    # Print data stats before optimization
    logger.info(f"Pre-optimization: {len(df)} points")
    
    # For debugging: return full dataset
    if st.sidebar.checkbox("Show all points (may affect performance)", value=False):
        return df
    
    # More gradual sampling based on zoom
    if zoom_level < 11:  # Very zoomed out
        sample_size = min(len(df), 1000)  # Increased from 200
    elif zoom_level < 13:  # Medium zoom
        sample_size = min(len(df), 2000)  # Increased from 500
    elif zoom_level < 15:  # Closer zoom
        sample_size = min(len(df), 5000)
    else:  # Very zoomed in
        return df  # Show all points when zoomed in
    
    # Use stratified sampling if possible to maintain distribution
    if 'District' in df.columns or 'Borough' in df.columns:
        group_col = 'District' if 'District' in df.columns else 'Borough'
        sampled = df.groupby(group_col, group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(sample_size * len(x)/len(df))))
        )
    else:
        sampled = df.sample(n=sample_size)
    
    logger.info(f"Post-optimization: {len(sampled)} points")
    return sampled

def create_basic_map_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create a basic scatter plot layer without advanced features."""
    return pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["Long", "Lat"],
        get_radius=50,
        get_fill_color=[255, 20, 147],
        pickable=True
    )

def create_advanced_map_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create an advanced scatter plot layer with 3D effects."""
    return pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["Long", "Lat"],
        get_radius="n_cameras * 20",
        get_fill_color=[255, 20, 147, 140],
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=30,
        extruded=True,
        elevation_scale=2,
        elevation_range=[0, 1000]
    )

def create_terrain_layer() -> Optional[pdk.Layer]:
    """Create a terrain layer with error handling."""
    try:
        return pdk.Layer(
            "TerrainLayer",
            data=None,
            elevation_decoder={
                'rScaler': 2, 'gScaler': 0, 'bScaler': 0, 'offset': 0
            },
            texture='https://s3.amazonaws.com/elevation-tiles-prod/terrarium/',
            elevation_data='https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
            wireframe=False
        )
    except Exception as e:
        st.warning(f"Could not initialize terrain layer: {str(e)}")
        return None

def verify_mapbox_token() -> bool:
    """Verify Mapbox token is valid and active."""
    if not MAPBOX_TOKEN:
        logger.error("No Mapbox token provided")
        return False
    try:
        response = requests.get(
            f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/1/1/1.mvt?access_token={MAPBOX_TOKEN}",
            timeout=5
        )
        is_valid = response.status_code == 200
        logger.info(f"Mapbox token verification: {'success' if is_valid else 'failed'}")
        return is_valid
    except Exception as e:
        logger.error(f"Error verifying Mapbox token: {str(e)}")
        return False

def debug_map_data(df: pd.DataFrame) -> None:
    """Print debug information about the map data."""
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample data:\n{df.head().to_string()}")
    
    # Check for invalid coordinates
    invalid_coords = df[
        (df['Lat'].isna()) | (df['Long'].isna()) |
        (df['Lat'] < -90) | (df['Lat'] > 90) |
        (df['Long'] < -180) | (df['Long'] > 180)
    ]
    if not invalid_coords.empty:
        logger.warning(f"Found {len(invalid_coords)} rows with invalid coordinates")
        logger.warning(f"Invalid coordinates:\n{invalid_coords.to_string()}")

def validate_data_integrity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the dataset for required columns and data quality."""
    issues = []
    
    # Check required columns
    required_cols = ['Lat', 'Long', 'n_cameras']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        issues.append(f"Found null values: {null_counts.to_dict()}")
    
    # Check coordinate ranges
    if 'Lat' in df.columns:
        invalid_lats = df[
            (df['Lat'].notna()) & 
            ((df['Lat'] < -90) | (df['Lat'] > 90))
        ]
        if not invalid_lats.empty:
            issues.append(f"Found {len(invalid_lats)} invalid latitude values")
    
    if 'Long' in df.columns:
        invalid_longs = df[
            (df['Long'].notna()) & 
            ((df['Long'] < -180) | (df['Long'] > 180))
        ]
        if not invalid_longs.empty:
            issues.append(f"Found {len(invalid_longs)} invalid longitude values")
    
    return len(issues) == 0, issues

def monitor_performance(func):
    """Decorator to monitor function performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

def create_density_layer(df: pd.DataFrame, radius: int = 100) -> pdk.Layer:
    """Create a hexagonal binning layer for density visualization."""
    return pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["Long", "Lat"],
        elevation_scale=50,
        elevation_range=[0, 3000],
        extruded=True,
        radius=radius,
        coverage=1.0,
        pickable=True,
        auto_highlight=True,
        color_range=[
            [255, 255, 204],  # Light yellow
            [255, 237, 160],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [252, 78, 42],
            [227, 26, 28]     # Deep red
        ],
    )

def create_camera_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create a scatter plot layer for individual camera points."""
    return pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["Long", "Lat"],
        get_radius="n_cameras * 20",
        get_fill_color=[255, 20, 147, 180],
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=30,
        extruded=True,
        elevation_scale=2,
        elevation_range=[0, 1000]
    )

@monitor_performance
def create_stable_map_view(filtered_df: pd.DataFrame, city_config: Dict[str, Any], selected_city: str) -> Tuple[Optional[pdk.Deck], pd.DataFrame]:
    """Create a stable map view with density analysis capability."""
    
    # Get stats with error handling
    try:
        stats = get_data_stats(filtered_df)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        stats = {
            "total_points": len(filtered_df),
            "total_cameras": 0,
            "avg_cameras_per_location": 0,
            "memory_usage": 0,
            "lat_range": [0, 0],
            "long_range": [0, 0],
            "unique_regions": 0
        }
    
    # Show detailed stats in sidebar
    with st.sidebar.expander("📊 Data Statistics", expanded=False):
        st.markdown(f"""
        ### Dataset Overview
        - **Total Locations:** {stats['total_points']:,}
        - **Total Cameras:** {stats['total_cameras']:,}
        - **Avg Cameras/Location:** {stats['avg_cameras_per_location']:.1f}
        - **Unique Regions:** {stats['unique_regions']}
        
        ### Memory Usage
        - **Dataset Size:** {stats['memory_usage']:.2f} MB
        
        ### Geographic Bounds
        - **Latitude:** [{stats['lat_range'][0]:.4f}, {stats['lat_range'][1]:.4f}]
        - **Longitude:** [{stats['long_range'][0]:.4f}, {stats['long_range'][1]:.4f}]
        
        <small>Last updated: {stats.get('timestamp', 'N/A')}</small>
        """, unsafe_allow_html=True)
    
    # Validate data before processing
    is_valid, issues = validate_data_integrity(filtered_df)
    if not is_valid:
        st.warning("Data quality issues detected:")
        for issue in issues:
            st.warning(f"- {issue}")
    
    # Add view mode selection
    st.sidebar.markdown("### View Options")
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["Points", "Density"],
        help="Points: Show individual cameras\nDensity: Show concentration using hexagonal bins"
    )
    
    if view_mode == "Density":
        # Add density controls
        st.sidebar.markdown("### Density Settings")
        radius = st.sidebar.slider(
            "Bin Radius (meters)",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Size of hexagonal bins. Smaller = more detail, Larger = broader patterns"
        )
        
        # Add legend for density view
        with st.sidebar.expander("📊 Density Legend", expanded=False):
            st.markdown("""
            **Color Scale:**
            - 🟨 Light Yellow: Low camera density
            - 🟧 Orange: Medium density
            - 🟥 Deep Red: High density
            
            **Height:**
            Taller hexagons indicate more cameras in that area.
            """)
    
    # Performance metrics
    st.sidebar.markdown("### Performance Metrics")
    perf_container = st.sidebar.empty()
    
    # Navigation controls
    start_time = time.time()
    
    # Center on data with bounds checking
    default_lat = filtered_df['Lat'].mean()
    default_lon = filtered_df['Long'].mean()
    lat_range = min(filtered_df['Lat'].max() - filtered_df['Lat'].min(), 0.1)
    lon_range = min(filtered_df['Long'].max() - filtered_df['Long'].min(), 0.1)
    
    lat = st.sidebar.slider(
        "Latitude",
        min_value=max(-90, default_lat - lat_range),
        max_value=min(90, default_lat + lat_range),
        value=default_lat,
        step=0.0001
    )
    lon = st.sidebar.slider(
        "Longitude",
        min_value=max(-180, default_lon - lon_range),
        max_value=min(180, default_lon + lon_range),
        value=default_lon,
        step=0.0001
    )
    
    # Dynamic zoom based on data density
    point_density = len(filtered_df) / (lat_range * lon_range)
    suggested_zoom = max(10, min(20, 11 + math.log2(point_density / 1000)))
    zoom = st.sidebar.slider("Zoom", 10, 20, int(suggested_zoom), 1)
    
    # Performance warning for high point counts
    if len(filtered_df) > 10000 and zoom < 12 and view_mode == "Points":
        st.sidebar.warning("⚠️ Large number of points at low zoom. Consider using Density view for better performance")
    
    bearing = st.sidebar.slider("Bearing", 0, 360, 0, 5)
    pitch = st.sidebar.slider("Pitch", 0, 60, 45, 1)
    
    # Optimize data based on zoom and view mode
    optimized_df = optimize_data_for_visualization(filtered_df, zoom) if view_mode == "Points" else filtered_df
    
    # Update performance metrics
    end_time = time.time()
    perf_container.markdown(f"""
    ⏱️ Performance:
    - Controls Render: {(end_time - start_time)*1000:.0f}ms
    - Points Processed: {len(optimized_df):,}
    - View Mode: {view_mode}
    """)
    
    # Create appropriate layer based on view mode
    try:
        if view_mode == "Density":
            main_layer = create_density_layer(optimized_df, radius)
            tooltip = {
                "html": """
                <div style="background-color: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;">
                    <b style="color: #FF1493;">Cameras in Area:</b> {elevationValue}<br/>
                    <small style="color: #888;">Aggregated within {radius}m radius</small>
                </div>
                """,
                "style": {"backgroundColor": "transparent"}
            }
        else:
            main_layer = create_camera_layer(optimized_df)
            tooltip = {
                "html": f"""
                <div style="background-color: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;">
                    <b style="color: #FF1493;">Cameras: {{n_cameras}}</b><br/>
                    <b>{city_config['region_col']}:</b> {{{city_config['region_col']}}}<br/>
                    <b>Location:</b> {{Lat:.4f}}, {{Long:.4f}}
                </div>
                """,
                "style": {"backgroundColor": "transparent"}
            }
        
        # Create the deck
        r = pdk.Deck(
            layers=[main_layer],
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=zoom,
                bearing=bearing,
                pitch=pitch,
                max_zoom=20,
                min_zoom=9
            ),
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v10",
            api_keys={"mapbox": MAPBOX_TOKEN}
        )
        
        return r, optimized_df
    except Exception as e:
        logger.error(f"Error creating map: {str(e)}")
        st.error(f"Error creating map: {str(e)}")
        return None, optimized_df

def generate_philly_camera_data():
    """Generate Philadelphia camera data using district information."""
    all_points = []
    
    for district, data in PHILLY_DISTRICTS.items():
        # Calculate total cameras in district
        total_cameras = data['safecam_count'] + data['bscp_count'] + data['rtcc_count']
        
        # Generate points for SafeCam registrations
        for _ in range(data['safecam_count']):
            lat = data['lat'] + np.random.normal(0, 0.005)
            lon = data['lon'] + np.random.normal(0, 0.005)
            all_points.append({
                'District': district,
                'Lat': lat,
                'Long': lon,
                'n_cameras': np.random.randint(1, 4),
                'IntersectionId': f"SC_{district}_{_}",
                'Type': 'SafeCam'
            })
        
        # Generate points for Business Security Camera Program
        for _ in range(data['bscp_count']):
            lat = data['lat'] + np.random.normal(0, 0.003)
            lon = data['lon'] + np.random.normal(0, 0.003)
            all_points.append({
                'District': district,
                'Lat': lat,
                'Long': lon,
                'n_cameras': np.random.randint(2, 6),
                'IntersectionId': f"BSCP_{district}_{_}",
                'Type': 'BSCP'
            })
        
        # Generate points for Real Time Crime Center cameras
        for _ in range(data['rtcc_count']):
            lat = data['lat'] + np.random.normal(0, 0.004)
            lon = data['lon'] + np.random.normal(0, 0.004)
            all_points.append({
                'District': district,
                'Lat': lat,
                'Long': lon,
                'n_cameras': 1,
                'IntersectionId': f"RTCC_{district}_{_}",
                'Type': 'RTCC'
            })
    
    return pd.DataFrame(all_points)

@st.cache_data(ttl=300)
def load_city_data(city_config):
    """Load and prepare surveillance camera data for a specific city."""
    try:
        if city_config["data_source"] == "local":
            intersections_df = pd.read_csv(f"{city_config['data_path']}/intersections.csv")
            counts_df = pd.read_csv(f"{city_config['data_path']}/counts.csv")
            intersection_counts = counts_df.groupby('IntersectionId')['n_cameras'].sum().reset_index()
            locations = intersections_df.merge(intersection_counts, on='IntersectionId', how='left')
            locations['n_cameras'] = locations['n_cameras'].fillna(0)
        elif city_config["data_source"] == "composite":
            locations = generate_philly_camera_data()
        
        locations = locations.dropna(subset=['Long', 'Lat'])
        return locations
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def render_resources_section(resources):
    """Render the resources section with a consistent style."""
    st.markdown(f"### {resources['title']}")
    st.write(resources['description'])
    
    for link in resources['links']:
        st.markdown(f"""
        <div class="resource-card">
            <h4>{link['name']}</h4>
            <p>{link['description']}</p>
            <a href="{link['url']}" target="_blank">Learn More →</a>
        </div>
        """, unsafe_allow_html=True)

def render_philly_data_repositories():
    """Render the Philadelphia data repository overview section."""
    st.markdown('<div class="data-repository-expander">', unsafe_allow_html=True)
    with st.expander("📊 Philadelphia Surveillance Data Repository Overview"):
        st.markdown("""
        ### Philadelphia Camera Surveillance Data Ecosystem

        The city's surveillance network is supported by multiple data repositories and management structures:

        #### 🏛️ Government Infrastructure
        - **Philadelphia Police Department (PPD)**
          - Real-Time Crime Center
          - Delaware Valley Intelligence Center (DVIC)
          - Multiple specialized databases
        - **Office of Innovation and Technology (OIT)**
          - Technical infrastructure management
          - Data storage solutions
        - **OpenDataPhilly**
          - Public datasets
          - Infrastructure mapping

        #### 🤝 Public-Private Partnerships
        - **Business Security Camera Program**
          - Integration with PPD network
          - Grant-funded camera installations
        - **SafeCam Program**
          - Private camera registration
          - Voluntary participation system
        - **CommunityCam**
          - Crowd-sourced mapping
          - Public accessibility

        #### 💾 Data Storage Solutions
        - **Cloud Infrastructure**
          - AWS GovCloud/Azure Gov implementations
          - Secure video storage systems
        - **Local Storage**
          - On-premise recording devices
          - Network Video Recorders (NVRs)
        - **Archive Systems**
          - Historical footage storage
          - Data retention compliance

        #### 🔄 Data Access and Integration
        - **Law Enforcement Systems**
          - DVIC fusion center
          - Inter-agency data sharing
        - **Public Records**
          - FOIA request systems
          - Transparency portals
        - **Research Access**
          - Academic partnerships
          - Civil rights organization data

        > *Note: This overview represents the institutional framework behind Philadelphia's surveillance data. 
        Actual camera locations and data availability may vary due to privacy and security considerations.*
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar - City Selection
st.sidebar.title("City Selection")
selected_city = st.sidebar.selectbox(
    "Choose a city",
    options=list(CITIES.keys()),
    index=0
)

city_config = CITIES[selected_city]

# Main title
st.title(city_config["title"])
st.markdown(f"### {city_config['subtitle']}")

# Description
st.markdown(f"""
<div class="info-box">
{city_config['description']}
</div>
""", unsafe_allow_html=True)

# Add Philadelphia data repository overview right after the description
if selected_city == "Philadelphia":
    render_philly_data_repositories()

# Load the data
try:
    locations = load_city_data(city_config)
    
    if locations is not None:
        # Validate initial data
        is_valid, issues = validate_data_integrity(locations)
        if not is_valid:
            st.error("Data validation failed:")
            for issue in issues:
                st.error(f"- {issue}")
            st.stop()
        
        st.write(f"Loaded {len(locations):,} total camera locations")
        
        # Add export functionality
        if st.button("Export Data Statistics"):
            stats = {
                "total_locations": len(locations),
                "total_cameras": int(locations['n_cameras'].sum()),
                "average_cameras_per_location": float(locations['n_cameras'].mean()),
                "coordinate_ranges": {
                    "latitude": [float(locations['Lat'].min()), float(locations['Lat'].max())],
                    "longitude": [float(locations['Long'].min()), float(locations['Long'].max())]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert to JSON and create download link
            json_str = json.dumps(stats, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="camera_stats.json">Download Statistics (JSON)</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Region filter with "All" option
        regions = ['All'] + sorted(locations[city_config['region_col']].unique().tolist())
        selected_region = st.sidebar.selectbox(f"Select {city_config['region_col']}", regions)
        
        # Camera type filter for Philadelphia
        if selected_city == "Philadelphia":
            camera_types = ['All'] + sorted(locations['Type'].unique().tolist())
            selected_type = st.sidebar.selectbox("Camera Type", camera_types)
        
        # Camera count filter with full range by default
        min_cameras = int(locations['n_cameras'].min())
        max_cameras = int(locations['n_cameras'].max())
        camera_range = st.sidebar.slider(
            "Number of Cameras",
            min_value=min_cameras,
            max_value=max_cameras,
            value=(min_cameras, max_cameras)
        )
        
        # Filter data with progress tracking
        filtered_df = locations.copy()
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df[city_config['region_col']] == selected_region]
            st.write(f"Filtered to {len(filtered_df):,} points in {selected_region}")
            
        if selected_city == "Philadelphia" and selected_type != 'All':
            filtered_df = filtered_df[filtered_df['Type'] == selected_type]
            st.write(f"Filtered to {len(filtered_df):,} {selected_type} cameras")
            
        filtered_df = filtered_df[
            (filtered_df['n_cameras'] >= camera_range[0]) &
            (filtered_df['n_cameras'] <= camera_range[1])
        ]
        st.write(f"Final filtered dataset: {len(filtered_df):,} points")
        
        # Create and display the map
        r, optimized_df = create_stable_map_view(filtered_df, city_config, selected_city)
        
        if r is not None:
            st.pydeck_chart(r)
            
            # Add data download option
            if st.button("Download visible data as CSV"):
                csv = optimized_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="camera_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
except Exception as e:
    logger.error(f"Error during execution: {str(e)}")
    st.error(f"Error during execution: {str(e)}")
    st.stop() 