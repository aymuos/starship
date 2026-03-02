# this is the polars rewrite of the features_trajectory.ipynb 

# install polars pyarrow 

import polars as pl
import pandas as pd

pl.Config.set_streaming_chunk_size(5_000_000)
pl.Config.set_tbl_rows(20)
dir_path = "../LaDe/"

#TODO: Change to parquet 

df_delivery = pd.read_pickle(f"{dir_path}/data_with_trajectory_20s/courier_detailed_trajectory_20s.pkl.xz", compression="xz")

# pl.from_pandas(df_delivery).write_parquet("../processed/temp/trajectory.parquet")   # THIS HAS TO BE lazy loading for asof join 

traj = pl.scan_parquet("../processed/temp/trajectory.parquet")
delivery = pl.scan_csv(f"{dir_path}/delivery_five_cities.csv")

print(traj.select(pl.all()).limit(3).collect())
print("Normalising the timestamps ")
# normalising the timestamps 

traj = traj.with_columns(
    pl.concat_str([pl.lit("2021-"), pl.col("gps_time")])
      .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
      .alias("gps_datetime")
)

delivery = delivery.with_columns([
    pl.concat_str([pl.lit("2021-"), pl.col("receipt_time")])
      .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
      .alias("receipt_datetime"),
    pl.concat_str([pl.lit("2021-"), pl.col("sign_time")])
      .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
      .alias("sign_datetime"),
])

print("Normalising the timestamps: DONE ")

print("Sorting trajectory ")
traj = (
    traj
    .sort(["postman_id", "gps_datetime"])
    .with_columns(
        pl.int_range(0, pl.len())
        .over("postman_id")
        .alias("idx")
    )
)  # IMP : idx = trajectory index per courier 

print("Sorting trajectory :DONE")

print("Calculating I and J : START")
# calculating i and j 
ij = (
    delivery
    .sort(["delivery_user_id", "receipt_datetime"])
    .join_asof(
        traj,
        left_on="receipt_datetime",
        right_on="gps_datetime",
        by_left="delivery_user_id",
        by_right="postman_id",
        strategy="forward",
    )
    .rename({"idx": "i"})
    .join_asof(
        traj,
        left_on="sign_datetime",
        right_on="gps_datetime",
        by_left="delivery_user_id",
        by_right="postman_id",
        strategy="backward",
    )
    .rename({"idx": "j"})
)

print("Calculating I and J : END")
print("Creating  time feats ")

# time feats 

ij = ij.with_columns([
    (
        pl.col("receipt_datetime").dt.hour() * 60 +
        pl.col("receipt_datetime").dt.minute() +
        pl.col("receipt_datetime").dt.second() / 60
    ).alias("accept_time_minute"),

    (
        pl.col("sign_datetime").dt.hour() * 60 +
        pl.col("sign_datetime").dt.minute() +
        pl.col("sign_datetime").dt.second() / 60
    ).alias("finish_time_minute"),
])

# delviery duration 

ij = ij.with_columns(
    (
        (pl.col("sign_datetime") - pl.col("receipt_datetime"))
        .dt.total_seconds() / 60
    ).alias("delivery_duration_min")
)

# distance to last package 

ij = ij.sort(["delivery_user_id", "ds", "sign_datetime"])

ij = ij.with_columns([
    pl.col("lng").shift(1).over(["delivery_user_id", "ds"]).alias("lng_prev"),
    pl.col("lat").shift(1).over(["delivery_user_id", "ds"]).alias("lat_prev"),
])

ij = ij.with_columns(
    pl.when(pl.col("lng_prev").is_null())
      .then(0.0)
      .otherwise(
          ((pl.col("lng") - pl.col("lng_prev"))**2 +
           (pl.col("lat") - pl.col("lat_prev"))**2).sqrt()
      )
      .alias("dis_to_last_package")
)


# TODO task features

ij = ij.with_row_index("order_idx")

todo = (
    ij
    .join(
        ij,
        on=["delivery_user_id", "ds"],
        how="inner",
        suffix="_b"
    )
    .filter(
        (pl.col("receipt_datetime_b") < pl.col("sign_datetime")) &
        (pl.col("order_idx_b") > pl.col("order_idx"))
    )
    .group_by("order_id")
    .agg(
        todo_task = pl.list("order_id_b"),
        todo_task_num = pl.len()
    )
)

ij = ij.join(todo, on="order_id", how="left")


# courier level aggregates 

courier_feat = (
    ij
    .group_by("delivery_user_id")
    .agg([
        pl.count().alias("order_sum"),
        pl.sum("dis_to_last_package").alias("dis_sum"),
        pl.n_unique("ds").alias("work_days"),
        pl.mean("delivery_duration_min").alias("time_avg_order"),
        pl.mean("dis_to_last_package").alias("dis_avg_order"),
    ])
    .with_columns([
        (pl.col("order_sum") / pl.col("work_days")).alias("order_avg_day"),
        (pl.col("dis_sum") / pl.col("work_days")).alias("dis_avg_day"),
    ])
)

ij = ij.join(courier_feat, on="delivery_user_id", how="left")

# GRID Features 

GRID = 20

ij = ij.with_columns([
    (pl.col("lng") / GRID).floor().cast(pl.Int32).alias("grid_x"),
    (pl.col("lat") / GRID).floor().cast(pl.Int32).alias("grid_y"),
]).with_columns(
    pl.concat_str([pl.col("grid_x"), pl.col("grid_y")], separator="_")
      .alias("grid_id")
)

# streaming output 
package_features = ij.collect(streaming=True)

package_features.write_parquet("package_feature.parquet")
courier_feat.collect(streaming=True).write_parquet("courier_feature.parquet")



## TO check 

# ij.select(
#     pl.col("delivery_duration_min").filter(pl.col("delivery_duration_min") < 0)
# ).collect()