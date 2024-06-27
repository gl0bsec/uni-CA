from google.cloud import storage

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # Initialize a storage client
    storage_client = storage.Client.from_service_account_json('hrtp.json')
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the file
    blob = bucket.blob(destination_blob_name)
    # Upload the file to Google Cloud Storage
    blob.upload_from_filename(source_file_name)

    return print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from the bucket."""
    storage_client = storage.Client.from_service_account_json('hrtp.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name}.")
