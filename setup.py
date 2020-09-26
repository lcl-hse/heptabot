import os
import subprocess
import nltk

from google.cloud import storage
from credentials import *


def dl_model(bucket_name, prefix, dl_dir='./models/savemodel/'):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    for blob in blobs:
        if not blob.name.endswith("/"):
            fname = dl_dir + blob.name[len(prefix):]
            bashCommand = "install -Dv /dev/null {}".format(fname)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            blob.download_to_filename(fname)  # Download

nltk.download("punkt")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(gcr_path)
dl_model(bucket_name, prefix)

