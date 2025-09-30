from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd
import ssl

app = FastAPI()
@app.get("/echs_data")
def echs_data(limit: int | None = None):
        # MongoDB connection string
    CONNECTION_STRING = "mongodb+srv://pilot:pilot@cluster1.rkupr.mongodb.net/?retryWrites=true&w=majority"

    # Connect to MongoDB
    client = MongoClient(CONNECTION_STRING)

    try:
        # Print available databases and collections
        print("Databases:", client.list_database_names())
        db = client.hospital_app
        print("Collections in hospital_app:", db.list_collection_names())

        # Load collections
        df = pd.DataFrame(list(db["ocr_results"].find()))
        df2 = pd.DataFrame(list(db["fs.files"].find()))
        df1 = pd.DataFrame(list(db["requests_history"].find()))

    finally:
        client.close()
        print("MongoDB connection closed.")
        print("", df.shape)
        print("", df2.shape)
        print("", df1.shape)
    df1['echs_card_result_id'] = df1['echs_card_result_id'].astype(str)
    df1['referral_letter_result_id'] = df1['referral_letter_result_id'].astype(str)
    df1['prescription_result_id'] = df1['prescription_result_id'].astype(str)
    df['_id'] = df['_id'].astype(str)
    main_df = df1[["echs_card_result_id","referral_letter_result_id","prescription_result_id"]].drop_duplicates()
    last_df = pd.merge(main_df, df, left_on='echs_card_result_id',right_on='_id', how='inner')
    last_df = last_df.rename(columns={'extracted_data': 'echs_data','doc_type':'echs_card'})
    last_df = pd.merge(last_df, df, left_on='referral_letter_result_id',right_on='_id', how='inner')
    last_df = last_df.rename(columns={'extracted_data': 'referral_letter_data','doc_type':'referral_letter','uploaded_at_x':'echs_upload_date','uploaded_at_y':'refferal_upload_date','image_file_id_x':'echs_image_id','image_file_id_y':'refferal_image_id',})
    final = last_df[['echs_image_id','echs_upload_date','echs_data','refferal_image_id'	,'referral_letter_data'	,'refferal_upload_date']]
    df3 = df2[['filename','_id']]
    final = pd.merge(final, df3, left_on='echs_image_id',right_on='_id', how='inner')
    final = pd.merge(final, df3, left_on='refferal_image_id',right_on='_id', how='inner')
    final = final.rename(columns = {'filename_x' : 'echs_img','filename_y' : 'refferal_img'})
    final = final[['echs_upload_date','echs_data','referral_letter_data'	,'refferal_upload_date','echs_img','refferal_img']]

    result = final.to_dict(orient='records')
    return {"echs_data": result}
