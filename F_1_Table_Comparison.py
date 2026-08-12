# ============================================================
# STEP 0 / STEP 1
# Freeze S1 and S2 at the same Delta point in time,
# check schema equality, and compare snapshot row counts.
#
# IMPORTANT:
# - timestampAsOf refers to the DELTA TABLE state at that time.
# - This is NOT a filter on the SCADA loadTimestamp column.
# - _s_source is added only to separate display/reporting copies.
# - _s_source must NOT be used later when testing row equality.
# ============================================================

from pyspark.sql import functions as F


# ------------------------------------------------------------
# 1. Fixed Delta comparison cutoff
#
# UTC timestamp with milliseconds and explicit timezone offset.
# This represents the end of 7 August 2026 UTC.
# ------------------------------------------------------------

COMPARISON_CUTOFF_TS = "2026-08-07T23:59:59.999+00:00"


# ------------------------------------------------------------
# 2. Source Delta tables
# ------------------------------------------------------------

s1_table = f"{S1_SCHEMA}.e_c_h"
s2_table = f"{S2_SCHEMA}.e_c_h"


# ------------------------------------------------------------
# 3. Read S1 Delta table exactly as it existed
#    at the comparison cutoff
# ------------------------------------------------------------

component_header_s1_snapshot_20260807 = (
    spark.read
    .option("timestampAsOf", COMPARISON_CUTOFF_TS)
    .table(s1_table)
)


# ------------------------------------------------------------
# 4. Read S2 Delta table exactly as it existed
#    at the SAME comparison cutoff
# ------------------------------------------------------------

component_header_s2_snapshot_20260807 = (
    spark.read
    .option("timestampAsOf", COMPARISON_CUTOFF_TS)
    .table(s2_table)
)


# ------------------------------------------------------------
# 5. Check schema equality BEFORE adding any helper columns
# ------------------------------------------------------------

schemas_identical = (
    component_header_s1_snapshot_20260807.schema
    ==
    component_header_s2_snapshot_20260807.schema
)

print("Comparison cutoff:", COMPARISON_CUTOFF_TS)
print("Schemas identical:", schemas_identical)


# ------------------------------------------------------------
# 6. If schemas differ, show both schemas for investigation
# ------------------------------------------------------------

if not schemas_identical:
    print("\nS1 schema:")
    component_header_s1_snapshot_20260807.printSchema()

    print("\nS2 schema:")
    component_header_s2_snapshot_20260807.printSchema()


# ------------------------------------------------------------
# 7. Count rows in the two frozen Delta snapshots
# ------------------------------------------------------------

s1_snapshot_count = component_header_s1_snapshot_20260807.count()
s2_snapshot_count = component_header_s2_snapshot_20260807.count()

snapshot_count_difference = s1_snapshot_count - s2_snapshot_count

print("\nS1 snapshot rows:", s1_snapshot_count)
print("S2 snapshot rows:", s2_snapshot_count)
print("S1 - S2 row-count difference:", snapshot_count_difference)
print("Row counts identical:", s1_snapshot_count == s2_snapshot_count)


# ------------------------------------------------------------
# 8. Optional helper:
#    create source-labelled copies for viewing/reporting.
#
#    DO NOT use _s_source later as part of row equality.
# ------------------------------------------------------------

def add_source_first(df, source_name):
    df_with_source = df.withColumn(
        "_s_source",
        F.lit(source_name)
    )

    ordered_cols = ["_s_source"] + df.columns

    return df_with_source.select(*ordered_cols)


component_header_s1_snapshot_20260807_with_source = add_source_first(
    component_header_s1_snapshot_20260807,
    "s1"
)

component_header_s2_snapshot_20260807_with_source = add_source_first(
    component_header_s2_snapshot_20260807,
    "s2"
)


# ------------------------------------------------------------
# 9. Final summary for this stage only
# ------------------------------------------------------------

print("\n============================================================")
print("FROZEN SNAPSHOT CHECK")
print("============================================================")
print(f"Cutoff                 : {COMPARISON_CUTOFF_TS}")
print(f"S1 table               : {s1_table}")
print(f"S2 table               : {s2_table}")
print(f"Schemas identical      : {schemas_identical}")
print(f"S1 rows                : {s1_snapshot_count}")
print(f"S2 rows                : {s2_snapshot_count}")
print(f"Row-count difference   : {snapshot_count_difference}")
print(
    f"Row counts identical   : "
    f"{s1_snapshot_count == s2_snapshot_count}"
)
print("============================================================")



# ------------------------------------------------------------
# STEP 2
# Compare the number of SCADA records per component_id
# at the frozen snapshot time.
#
# This is a diagnostic step only.
# Matching counts do NOT yet prove that the underlying
# records are identical.
# ------------------------------------------------------------

COMPONENT_ID_COL = "component_id"


# ------------------------------------------------------------
# Count records per component_id in the frozen S1 snapshot
# ------------------------------------------------------------

s1_component_counts_at_T = (
    component_header_s1_snapshot_20260807
    .groupBy(COMPONENT_ID_COL)
    .count()
    .withColumnRenamed("count", "s1_count")
)


# ------------------------------------------------------------
# Count records per component_id in the frozen S2 snapshot
# ------------------------------------------------------------

s2_component_counts_at_T = (
    component_header_s2_snapshot_20260807
    .groupBy(COMPONENT_ID_COL)
    .count()
    .withColumnRenamed("count", "s2_count")
)


# ------------------------------------------------------------
# Full join the component counts so that components present
# on only one server are also retained
# ------------------------------------------------------------

component_count_comparison_at_T = (
    s1_component_counts_at_T
    .join(
        s2_component_counts_at_T,
        on=COMPONENT_ID_COL,
        how="full"
    )
    .fillna(
        0,
        subset=["s1_count", "s2_count"]
    )
)


# ------------------------------------------------------------
# Calculate S1 minus S2 record-count difference
#
# Negative value:
# S1 has fewer records for that component.
#
# Positive value:
# S1 has more records for that component.
# ------------------------------------------------------------

component_count_comparison_at_T = (
    component_count_comparison_at_T
    .withColumn(
        "count_difference",
        F.col("s1_count") - F.col("s2_count")
    )
)


# ------------------------------------------------------------
# Keep only component_ids where the record counts differ
# ------------------------------------------------------------

component_count_differences_at_T = (
    component_count_comparison_at_T
    .filter(
        F.col("count_difference") != 0
    )
    .orderBy(
        F.abs(F.col("count_difference")).desc(),
        F.col(COMPONENT_ID_COL)
    )
)


# ------------------------------------------------------------
# Count how many component_ids have different record counts
# ------------------------------------------------------------

different_component_count = (
    component_count_differences_at_T
    .count()
)


# ------------------------------------------------------------
# Check that the component-level differences reconcile
# back to the total S1 - S2 row-count difference
# ------------------------------------------------------------

component_difference_total = (
    component_count_differences_at_T
    .agg(
        F.sum("count_difference").alias("difference_total")
    )
    .first()["difference_total"]
)


# ------------------------------------------------------------
# Display the component_ids where S1 and S2 differ
# ------------------------------------------------------------

display(
    component_count_differences_at_T
)


# ------------------------------------------------------------
# Summary for this step
# ------------------------------------------------------------

print("\n============================================================")
print("COMPONENT RECORD-COUNT COMPARISON AT FROZEN TIME")
print("============================================================")
print("Comparison cutoff                :", COMPARISON_CUTOFF_TS)
print("Component IDs with count mismatch:", different_component_count)
print("S1 - S2 difference from Step 1   :", snapshot_count_difference)
print("Sum of component differences     :", component_difference_total)
print(
    "Difference totals reconcile     :",
    component_difference_total == snapshot_count_difference
)
print("============================================================")
