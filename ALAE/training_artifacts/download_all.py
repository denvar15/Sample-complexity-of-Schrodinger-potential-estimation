import os
import gdown
import urllib.request


def download_from_google_drive(file_id, destination_dir, filename=None):
    os.makedirs(destination_dir, exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(destination_dir, filename if filename else file_id)
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        raise IOError(f"Failed to download from Google Drive: {file_id}") from e


def download_from_url(url, destination_dir, filename=None):
    os.makedirs(destination_dir, exist_ok=True)
    output = os.path.join(destination_dir, filename if filename else os.path.basename(url))
    try:
        urllib.request.urlretrieve(url, output)
    except Exception as e:
        raise IOError(f"Failed to download from URL: {url}") from e


# FFHQ
try:
    download_from_google_drive('170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz', 'training_artifacts/ffhq', 'model_submitted.pth')
    download_from_google_drive('1QESywJW8N-g3n0Csy0clztuJV99g8pRm', 'training_artifacts/ffhq', 'model_194.pth')
    download_from_google_drive('18BzFYKS3icFd1DQKKTeje7CKbEKXPVug', 'training_artifacts/ffhq', 'model_157.pth')
except IOError:
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth', 'training_artifacts/ffhq')
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_194.pth', 'training_artifacts/ffhq')
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth', 'training_artifacts/ffhq')
