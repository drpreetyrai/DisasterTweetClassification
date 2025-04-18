import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# AWS S3 settings
bucket_name = "mlops-44448888"  # Your S3 bucket
local_path = 'tinybert-disaster-tweet'  # Local directory to save model
s3_prefix = 'ml-models/tinybert-disaster-tweet'  # S3 path where model is stored

# Initialize S3 client
s3 = boto3.client('s3')

# Function to download model directory from S3
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                if s3_key.endswith('/'):
                    continue  # skip folders
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

# Streamlit UI
st.title("🚨 Disaster Tweet Classification App")

# Button to download the model
button = st.button("⬇️ Download Model from S3")
if button:
    with st.spinner("Downloading model files from S3... Please wait!"):
        download_dir(local_path, s3_prefix)
    st.success("Model Downloaded Successfully!")

# Text input for the user
text = st.text_area("Enter a Tweet to Classify:", "Type here...")

# Predict button
predict = st.button("🚀 Predict Disaster or Not")

# Load model only when needed
if predict:
    with st.spinner("Loading model and predicting..."):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        classifier = pipeline('text-classification', model=local_path, device=0 if torch.cuda.is_available() else -1)
        
        output = classifier(text)
        label = output[0]['label']
        score = output[0]['score']

        # Display
        st.subheader("Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {score:.4f}")

        if label.lower() == 'disaster':
            st.error("🔴 Disaster-related Tweet Detected!")
        else:
            st.success("🟢 Not a Disaster-related Tweet.")






