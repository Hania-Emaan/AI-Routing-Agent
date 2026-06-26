from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# 1. Initialize FastAPI Application
app = FastAPI(title="Pakistan Intercity Transport AI Routing Engine", version="1.0")

# Enable CORS so your frontend web browser UI can safely make API calls to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all web interfaces to connect during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Assert and Load Core AI Resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_PATH = os.path.join(BASE_DIR, "pk.csv")
MODEL_PATH = os.path.join(BASE_DIR, "pakistan_intercity_agent.json")

if not os.path.exists(GEO_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("CRITICAL ERROR: Ensure 'pk.csv' and 'pakistan_intercity_agent.json' are in the same folder as main.py!")

# Load static layout map
geo_df = pd.read_csv(GEO_PATH)

# Load machine learning model brain
agent_brain = xgb.XGBRegressor()
agent_brain.load_model(MODEL_PATH)

# Reference lists for lookups and validation
MAJOR_CITIES = ['Faisalabad', 'Islamabad', 'Karachi', 'Lahore', 'Multan', 'Peshawar', 'Quetta', 'Rawalpindi', 'Sargodha', 'Sialkot City']
SEASONS = ['Autumn', 'Spring', 'Summer', 'Winter']
HIGHWAYS = ['Motorway', 'National_Highway']
RAINFALLS = ['heavy', 'light', 'moderate', 'none']

# Hardcoded feature columns matching the model's precise training alignment matrix
FEATURE_COLUMNS = [
    'distance_km', 'month', 'average_temp', 'humidity',
    'origin_city_Faisalabad', 'origin_city_Islamabad', 'origin_city_Karachi', 'origin_city_Lahore',
    'origin_city_Multan', 'origin_city_Peshawar', 'origin_city_Quetta', 'origin_city_Rawalpindi',
    'origin_city_Sargodha', 'origin_city_Sialkot City',
    'destination_city_Faisalabad', 'destination_city_Islamabad', 'destination_city_Karachi', 'destination_city_Lahore',
    'destination_city_Multan', 'destination_city_Peshawar', 'destination_city_Quetta', 'destination_city_Rawalpindi',
    'destination_city_Sargodha', 'destination_city_Sialkot City',
    'highway_type_Motorway', 'highway_type_National_Highway',
    'season_Autumn', 'season_Spring', 'season_Summer', 'season_Winter',
    'rainfall_intensity_heavy', 'rainfall_intensity_light', 'rainfall_intensity_moderate', 'rainfall_intensity_none'
]


# 3. Define the Structured Input Schema (Pydantic Validation Layer)
class RouteRequest(BaseModel):
    origin_city: str
    destination_city: str
    highway_type: str  # 'Motorway' or 'National_Highway'
    month: int
    season: str        # 'Winter', 'Summer', 'Autumn', 'Spring'
    average_temp: float
    humidity: float
    rainfall_intensity: str  # 'none', 'light', 'moderate', 'heavy'


# 4. Mathematical Geometry Helper (Haversine Formula)
def calculate_road_distance(origin: str, dest: str) -> float:
    try:
        c1 = geo_df[geo_df['city'] == origin].iloc[0]
        c2 = geo_df[geo_df['city'] == dest].iloc[0]
        
        R = 6371.0 # Earth's radius in KM
        lat1, lon1, lat2, lon2 = map(np.radians, [c1['lat'], c1['lng'], c2['lat'], c2['lng']])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c * 1.25 # Accounts for real highway route curves
    except IndexError:
        raise HTTPException(status_code=400, detail=f"City routing geometry unmapped. Choose from: {MAJOR_CITIES}")


# 5. API Route Endpoints
@app.get("/")
def health_check():
    return {"status": "online", "agent_brain_loaded": True, "active_hubs": MAJOR_CITIES}


@app.post("/api/predict-route")
def predict_route(request: RouteRequest):
    # Step A: Validate matching intercity parameters
    if request.origin_city == request.destination_city:
        raise HTTPException(status_code=400, detail="Origin and Destination cannot be identical.")

    # Step B: Compute actual geolocated road distance
    distance_km = calculate_road_distance(request.origin_city, request.destination_city)

    # Step C: Formulate blank structural input array matching training columns
    input_vector = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
    
    # Step D: Map numerical feature values
    input_vector['distance_km'] = distance_km
    input_vector['month'] = request.month
    input_vector['average_temp'] = request.average_temp
    input_vector['humidity'] = request.humidity

    # Step E: Trigger One-Hot binary flags to 1 based on request inputs
    if f'origin_city_{request.origin_city}' in FEATURE_COLUMNS:
        input_vector[f'origin_city_{request.origin_city}'] = 1
    if f'destination_city_{request.destination_city}' in FEATURE_COLUMNS:
        input_vector[f'destination_city_{request.destination_city}'] = 1
    if f'highway_type_{request.highway_type}' in FEATURE_COLUMNS:
        input_vector[f'highway_type_{request.highway_type}'] = 1
    if f'season_{request.season}' in FEATURE_COLUMNS:
        input_vector[f'season_{request.season}'] = 1
    if f'rainfall_intensity_{request.rainfall_intensity}' in FEATURE_COLUMNS:
        input_vector[f'rainfall_intensity_{request.rainfall_intensity}'] = 1

    # Step F: Run the inputs through the XGBoost Model weights
    try:
        predicted_minutes = float(agent_brain.predict(input_vector)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Failure: {str(e)}")

    # Step G: Return clean structured response back to client frontend
    return {
        "origin": request.origin_city,
        "destination": request.destination_city,
        "calculated_distance_km": round(distance_km, 2),
        "highway_infrastructure": request.highway_type,
        "environmental_conditions": {
            "season": request.season,
            "temperature_c": request.average_temp,
            "humidity_percent": request.humidity,
            "rain_intensity": request.rainfall_intensity
        },
        "predicted_travel_time": {
            "total_minutes": round(predicted_minutes, 2),
            "formatted_hours": round(predicted_minutes / 60, 2)
        }
    }