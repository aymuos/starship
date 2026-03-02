# LaDe dataset what impacts ETA most 

## roads.csv 

- geometry → Provides the actual shape and coordinates of the road segment. Combined with osm_id, this lets you calculate distance and align courier GPS traces to roads.

- length (derived from geometry) → Direct input for travel time = distance ÷ speed.

- maxspeed → Critical for estimating baseline travel speed on each segment = 17 distinct values 

- oneway → Determines whether a courier can legally traverse the road in both directions. Essential for route feasibility. 

- fclass (functional class) → Road hierarchy (motorway vs. residential). Helps model expected speeds and congestion patterns.

- layer, bridge, tunnel → Useful for network topology (e.g., flyovers vs. tunnels affect connectivity and traffic flow).

- city → Needed for context — traffic patterns vary by city (Shanghai vs. Hangzhou).

- ref → Highway codes can help group roads (e.g., expressways vs. local streets).


The geometry field in LaDe’s roads.csv is essentially a polyline (a sequence of latitude/longitude coordinates).
We can compute the length of the road segment directly using standard geospatial libraries.


- geometry == LINESTRING(120.123 30.456, 120.124 30.457, ...)

calculating dist = Apply the Haversine formula or use pyproj.Geod to calculate the real-world distance between successive points.
---------------------------------------------------------------------------------------------------------------------------------------------------

# Delivery
```delivery_five_cities.csv```

Purpose: order-level ground truth + POI / AOI metadata
Used for:

AOI-based aggregation

# courier_detailed_trajectory.csv

Purpose: behavioral signals
Used for: congestion,dwell time,fatigue,familiarity ,turn ratios

ds	| postman_id	| gps_time	| lat	| lng
318	01890dd2fdc077b8deb7d8c120bf9c9f	03-18 00:01:04	2.375271e+06	1.566731e+06

# roads.csv
