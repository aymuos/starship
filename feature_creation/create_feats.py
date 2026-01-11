## Objective : create new features 

# T0 = accept_time
# T1 = delivery_time

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2




orders = pd.read_csv("../LaDe/delivery/delivery_cq.csv", parse_dates=[
    "accept_time", "accept_gps_time",
    "delivery_time", "delivery_gps_time"
])

traj = pd.read_csv("../LaDe/courier_detailed_trajectory/courier_detailed_trajectory/.csv", parse_dates=["gps_time"])
roads = pd.read_csv("../LaDe/road-network/roads.csv")


def haversine(lat1, lon1, lat2, lon2):
    ''' DIstance between 2 lat long on a spherical surface '''
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# Segment Velocity (Congestion Proxy)

traj = traj.sort_values(["postman_id", "gps_time"])

traj["dt"] = traj.groupby("postman_id")["gps_time"].diff().dt.total_seconds()
traj["dist"] = haversine(
    traj["lat"].shift(), traj["lng"].shift(),
    traj["lat"], traj["lng"]
)

traj["v_seg"] = traj["dist"] / traj["dt"]
traj = traj[(traj["v_seg"] > 0) & (traj["v_seg"] < 40)] 

traj["hour"] = traj["gps_time"].dt.hour()

city_speed = (
    traj.groupby(["city", "hour"])
    .v_seg.median()
    .reset_index(name="city_speed")
)

# Driver Familiarity (Region Visit Count)
# Strong (causal)

orders = orders.sort_values("accept_time")

orders["N_visit"] = (
    orders
    .groupby(["courier_id", "region_id"])
    .cumcount()
)

# Dwell Time (Parking Difficulty)

traj["speed"] = traj["v_seg"]

stops = traj[traj["speed"] < 0.5]

dwell = (
    stops.groupby(["postman_id", "ds"])
    .dt.sum()
    .reset_index(name="T_dwell")
)

avg_dwell = (
    dwell.groupby("region_id")
    .T_dwell.mean()
    .reset_index(name="avg_dwell_region")
)

# v_seg already computed earlier
STATIONARY_THRESHOLD = 0.5  # m/s (~1.8 km/h)

traj["is_stationary"] = traj["v_seg"] < STATIONARY_THRESHOLD



