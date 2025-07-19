import os
import json
import decimal # Import the decimal library to check for the type
from google.cloud import bigquery, firestore
from bigquery_utils import get_master_item_analytics,get_peak_shopping_times, get_average_repurchase_time,convert_row_to_dict,init_bigquery_client
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

try:
    client, db, TABLE = init_bigquery_client()
except ValueError as e:
    logger.error(f"Initialization failed: {e}")
    exit(1)

def get_analytics() -> str:
    """
    Executes all analytics queries and saves their results to a single
    'market_analytics' document in Firestore.
    """
    try:
        logger.info("Starting analytics queries...")
        
        item_stats = [convert_row_to_dict(row) for row in get_master_item_analytics(client, TABLE)]
        logger.info(f"Query 1 (Item Analytics) completed. {len(item_stats)} items found.")

        peak_times = [convert_row_to_dict(row) for row in get_peak_shopping_times(client, TABLE)]
        logger.info(f"Query 2 (Peak Shopping Times) completed. {len(peak_times)} records found.")

        repurchase_rates = [convert_row_to_dict(row) for row in get_average_repurchase_time(client, TABLE)]
        logger.info(f"Query 3 (Average Repurchase Time) completed. {len(repurchase_rates)} items found.")
        
        analytics_data = {
            "master_item_analytics": item_stats,
            "peak_shopping_times": peak_times,
            "average_repurchase_rates": repurchase_rates,
            "last_updated": firestore.SERVER_TIMESTAMP
        }

        doc_ref = db.collection("market_analytics").document("latest_run")
        doc_ref.set(analytics_data)
        
        logger.info("Successfully saved full analytics report to Firestore.")
        return "Pipeline completed successfully."

    except Exception as e:
        logger.exception("An error occurred in the analytics pipeline.") 
        raise e 

if __name__ == "__main__":
    try:
        result = get_analytics()
        logger.info(result)
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        