import boto3
import json
import os
from botocore.config import Config

# ------------------------------------------------------------
# CONFIGURATION - UPDATE THESE VALUES
# ------------------------------------------------------------

KNOWLEDGE_BASE_ID = "YCXJT4XOV3"   # <-- Replace if needed
S3_BUCKET_NAME = "bedrock-kb-255401567831"
S3_PREFIX = "spec-sheets/"  # Folder where PDFs are uploaded

# ------------------------------------------------------------

client = boto3.client(
    "bedrock-agent",
    region_name="us-west-2",
    config=Config(retries={"max_attempts": 10})
)

def list_s3_files():
    """List all PDFs inside the S3 folder."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)

    if "Contents" not in response:
        print("No PDF files found in S3.")
        return []

    files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".pdf")]
    return files

def ingest_file_to_kb(file_key):
    """Send a document to the knowledge base for ingestion."""
    print(f"\n📌 Sending PDF to Knowledge Base: {file_key}")

    response = client.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId="s3",  # Default S3 datasource created in Terraform
        ingestionJobConfig={
            "s3IngestionConfig": {
                "bucketArn": f"arn:aws:s3:::{S3_BUCKET_NAME}",
                "prefix": S3_PREFIX
            }
        }
    )

    job_id = response["ingestionJob"]["ingestionJobId"]
    print(f"🚀 Ingestion started. Job ID: {job_id}")
    return job_id

def main():
    print("\n🔍 Looking for PDF files in S3...")

    files = list_s3_files()
    if not files:
        print("❌ No files found. Upload PDFs first.")
        return

    print(f"✅ Found {len(files)} file(s):")
    for f in files:
        print(" -", f)

    # Trigger ingestion
    ingest_file_to_kb(files[0])

    print("\n🎉 Ingestion triggered successfully!")
    print("⚠️ It will take a few minutes for embeddings to be generated.")

if __name__ == "__main__":
    main()
