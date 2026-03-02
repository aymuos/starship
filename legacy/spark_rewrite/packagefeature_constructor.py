import polars as pl

# =========================================================
# helpers
# =========================================================
def euclid(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2).sqrt()


# =========================================================
# 1) LOAD DATASETS
# =========================================================

print("Loading orders")

orders = (
    pl.scan_csv("../LaDe/delivery_five_cities.csv")
    .select([
        "order_id",
        pl.col("delivery_user_id").alias("courier_id"),
        "poi_lat","poi_lng",
        "receipt_lat","receipt_lng",
        "sign_lat","sign_lng",
        "ds",

        (pl.lit("2021-")+pl.col("receipt_time"))
            .str.strptime(pl.Datetime,"%Y-%m-%d %H:%M:%S",strict=False)
            .alias("receipt_time"),

        (pl.lit("2021-")+pl.col("sign_time"))
            .str.strptime(pl.Datetime,"%Y-%m-%d %H:%M:%S",strict=False)
            .alias("sign_time"),
    ])
    .sort(["courier_id","receipt_time"])
)

print("Loading trajectory")

traj = (
    pl.scan_parquet("../processed/temp/trajectory.parquet")
    .select([
        pl.col("postman_id").alias("courier_id"),
        "lat","lng","ds",
        (pl.lit("2021-")+pl.col("gps_time"))
            .str.strptime(pl.Datetime,"%Y-%m-%d %H:%M:%S",strict=False)
            .alias("gps_time")
    ])
    .sort(["courier_id","gps_time"])
)

# =========================================================
# 2) TEMPORAL INTERVAL JOIN (CRITICAL FIX)
# =========================================================

print("Matching GPS points to active orders")

order_start = orders.select([
    "order_id","courier_id","receipt_time",
    "sign_time","poi_lat","poi_lng",
    "receipt_lat","receipt_lng"
])

traj_orders = (
    traj.join_asof(
        order_start,
        left_on="gps_time",
        right_on="receipt_time",
        by="courier_id",
        strategy="backward"
    )
    .filter(pl.col("gps_time") <= pl.col("sign_time"))
)

# =========================================================
# 3) MOTION FEATURES
# =========================================================

print("Computing motion features")

traj_feat = (
    traj_orders
    .with_columns([
        pl.col("lat").shift(-1).over("order_id").alias("lat2"),
        pl.col("lng").shift(-1).over("order_id").alias("lng2"),
        pl.col("gps_time").shift(-1).over("order_id").alias("t2"),
    ])
    .with_columns([
        euclid(pl.col("lng"),pl.col("lat"),pl.col("lng2"),pl.col("lat2")).alias("step_dist"),
        (pl.col("t2")-pl.col("gps_time")).dt.total_seconds().alias("step_time")
    ])
)

mobility = (
    traj_feat.group_by("order_id")
    .agg([
        pl.sum("step_dist").alias("path_length"),
        pl.sum("step_time").alias("travel_seconds"),
        (pl.sum("step_dist")/pl.sum("step_time")).alias("avg_speed"),
        (pl.col("step_dist")<20).mean().alias("stop_ratio"),
        pl.len().alias("num_points")
    ])
)

# =========================================================
# 4) DIRECT DISTANCE + DETOUR
# =========================================================

print("Computing detour")

geo = (
    orders.select([
        "order_id","receipt_lat","receipt_lng","poi_lat","poi_lng"
    ])
    .with_columns(
        euclid(pl.col("receipt_lng"),pl.col("receipt_lat"),
               pl.col("poi_lng"),pl.col("poi_lat")).alias("direct_dist")
    )
)

features = (
    mobility.join(geo,on="order_id",how="left")
    .with_columns((pl.col("path_length")/pl.col("direct_dist")).alias("detour_ratio"))
)

# =========================================================
# 5) SERVICE DWELL TIME
# =========================================================

print("Computing service time")

arrival = (
    traj_orders
    .with_columns(
        euclid(pl.col("lng"),pl.col("lat"),
               pl.col("poi_lng"),pl.col("poi_lat")).alias("dist_to_dest")
    )
    .filter(pl.col("dist_to_dest")<80)
    .group_by("order_id")
    .agg(pl.min("gps_time").alias("arrival_time"))
)

service = (
    arrival.join(orders.select(["order_id","sign_time"]),on="order_id")
    .with_columns((pl.col("sign_time")-pl.col("arrival_time")).dt.total_seconds().alias("service_time"))
)

# =========================================================
# 6) TEMPORAL WORKLOAD W(t)
# =========================================================

print("Computing workload process")

events = pl.concat([
    orders.select(["courier_id",pl.col("receipt_time").alias("time"),pl.lit(1).alias("delta")]),
    orders.select(["courier_id",pl.col("sign_time").alias("time"),pl.lit(-1).alias("delta")])
]).sort(["courier_id","time"])

workload_process = events.with_columns(
    pl.col("delta").cum_sum().over("courier_id").alias("workload")
)

traj_with_load = (
    traj_orders.join_asof(
        workload_process,
        left_on="gps_time",
        right_on="time",
        by="courier_id",
        strategy="backward"
    )
)

workload_features = (
    traj_with_load
    .with_columns([
        ((pl.col("gps_time")-pl.col("receipt_time")).dt.total_seconds() /
         (pl.col("sign_time")-pl.col("receipt_time")).dt.total_seconds()).alias("progress")
    ])
    .group_by("order_id")
    .agg([
        pl.col("workload").mean().alias("avg_workload"),
        pl.col("workload").max().alias("peak_workload"),
        pl.when(pl.col("progress")<0.3).then(pl.col("workload")).otherwise(None).mean().alias("early_load"),
        pl.when((pl.col("progress")>=0.3)&(pl.col("progress")<0.7)).then(pl.col("workload")).otherwise(None).mean().alias("mid_load"),
        pl.when(pl.col("progress")>=0.7).then(pl.col("workload")).otherwise(None).mean().alias("late_load"),
    ])
)

# =========================================================
# 7) FINAL TABLE
# =========================================================

print("Building final dataset")

final = (
    orders.select(["order_id","receipt_time","sign_time"])
    .with_columns((pl.col("sign_time")-pl.col("receipt_time")).dt.total_seconds().alias("eta"))
    .join(features,on="order_id",how="left")
    .join(service,on="order_id",how="left")
    .join(workload_features,on="order_id",how="left")
)

# =========================================================
# EXECUTE
# =========================================================

print(final.head(10).collect())
final.collect().write_parquet("lade_features.parquet")

print("DONE")
