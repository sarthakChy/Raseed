import os
import json
import decimal
from google.cloud import bigquery, firestore

# This is the base CTE (Common Table Expression) query. It gets the latest version of each receipt.
LATEST_RECEIPTS_CTE = """
WITH LatestReceipts AS (
    SELECT
      ARRAY_AGG(t ORDER BY timestamp DESC LIMIT 1)[OFFSET(0)] AS record
    FROM
      `{TABLE}` AS t
    GROUP BY
      document_id
)
"""

def get_master_item_analytics(client: bigquery.Client, TABLE: str) -> list:
    """
    Query 1: Calculates count, avg price, most popular store,
    and the cheapest store for every unique item.
    """
    query = f"""
        {LATEST_RECEIPTS_CTE}
        , ItemPricesByMerchant AS (
            SELECT
              JSON_EXTRACT_SCALAR(record.data, '$.merchantName') AS merchant_name,
              JSON_EXTRACT_SCALAR(item, '$.name') AS item_name,
              CAST(JSON_EXTRACT_SCALAR(item, '$.price') AS NUMERIC) AS item_price
            FROM LatestReceipts, UNNEST(JSON_EXTRACT_ARRAY(record.data, '$.items')) AS item
            WHERE JSON_EXTRACT_SCALAR(item, '$.name') IS NOT NULL AND CAST(JSON_EXTRACT_SCALAR(item, '$.price') AS NUMERIC) > 0
        ),
        AggregatedStats AS (
            SELECT
                item_name,
                merchant_name,
                COUNT(*) as purchase_count,
                AVG(item_price) as avg_price
            FROM ItemPricesByMerchant
            GROUP BY item_name, merchant_name
        )
        SELECT
            item_name,
            SUM(purchase_count) as total_purchase_count,
            ROUND(AVG(avg_price), 2) as overall_average_price,
            ARRAY_AGG(STRUCT(merchant_name, purchase_count) ORDER BY purchase_count DESC LIMIT 1)[OFFSET(0)].merchant_name AS most_popular_store,
            ARRAY_AGG(STRUCT(merchant_name, avg_price) ORDER BY avg_price ASC LIMIT 1)[OFFSET(0)].merchant_name AS cheapest_store,
            ROUND(ARRAY_AGG(STRUCT(merchant_name, avg_price) ORDER BY avg_price ASC LIMIT 1)[OFFSET(0)].avg_price, 2) AS lowest_average_price
        FROM AggregatedStats
        GROUP BY item_name
        ORDER BY total_purchase_count DESC;
    """
    return list(client.query(query.format(TABLE=TABLE)).result())

def get_peak_shopping_times(client: bigquery.Client, TABLE: str) -> list:
    """
    Query 2: Finds the busiest days of the week and hours for shopping.
    """
    query = f"""
        {LATEST_RECEIPTS_CTE}
        , TransactionDates AS (
            SELECT
                TIMESTAMP(JSON_EXTRACT_SCALAR(record.data, '$.transactionDate')) AS transaction_timestamp,
                record.document_id
            FROM LatestReceipts
            WHERE JSON_EXTRACT_SCALAR(record.data, '$.transactionDate') IS NOT NULL
        )
        SELECT
            EXTRACT(DAYOFWEEK FROM transaction_timestamp AT TIME ZONE 'Asia/Kolkata') as day_of_week_code,
            FORMAT_TIMESTAMP('%A', transaction_timestamp, 'Asia/Kolkata') as day_of_week_name,
            EXTRACT(HOUR FROM transaction_timestamp AT TIME ZONE 'Asia/Kolkata') as hour_of_day,
            COUNT(document_id) as transaction_count
        FROM TransactionDates
        GROUP BY day_of_week_code, day_of_week_name, hour_of_day
        ORDER BY transaction_count DESC;
    """
    return list(client.query(query.format(TABLE=TABLE)).result())

def get_average_repurchase_time(client: bigquery.Client, TABLE: str) -> list:
    """
    Query 3: Calculates the average time between purchases for items.
    """
    query = f"""
        {LATEST_RECEIPTS_CTE}
        , UserPurchases AS (
            SELECT
                JSON_EXTRACT_SCALAR(record.data, '$.userId') AS userId,
                JSON_EXTRACT_SCALAR(item, '$.name') AS item_name,
                TIMESTAMP(JSON_EXTRACT_SCALAR(record.data, '$.transactionDate')) AS purchase_date
            FROM LatestReceipts, UNNEST(JSON_EXTRACT_ARRAY(record.data, '$.items')) AS item
            WHERE JSON_EXTRACT_SCALAR(record.data, '$.transactionDate') IS NOT NULL
        ),
        PurchaseIntervals AS (
            SELECT
                item_name,
                purchase_date,
                LAG(purchase_date, 1) OVER (PARTITION BY userId, item_name ORDER BY purchase_date) AS previous_purchase_date
            FROM UserPurchases
        )
        SELECT
            item_name,
            AVG(TIMESTAMP_DIFF(purchase_date, previous_purchase_date, DAY)) as avg_days_between_purchases
        FROM PurchaseIntervals
        WHERE previous_purchase_date IS NOT NULL
        GROUP BY item_name
        HAVING COUNT(item_name) > 1; -- Only for items bought at least twice
    """
    return list(client.query(query.format(TABLE=TABLE)).result())


def convert_row_to_dict(row):
    """Helper function to convert a BigQuery Row to a dict, casting Decimals to floats."""
    row_dict = dict(row)
    for key, value in row_dict.items():
        if isinstance(value, decimal.Decimal):
            row_dict[key] = float(value)
    return row_dict

def init_bigquery_client():
    project_id = os.getenv("GCP_PROJECT_ID")
    bigquery_table = os.getenv("TABLE")

    print(f"Initializing BigQuery client for project: {project_id}, table: {bigquery_table}")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable not set.")

    if not bigquery_table:
        raise ValueError("TABLE environment variable not set.")    

    client = bigquery.Client(project=project_id)
    db = firestore.Client(project=project_id)
    TABLE = f"{project_id}.{bigquery_table}"

    return client, db, TABLE