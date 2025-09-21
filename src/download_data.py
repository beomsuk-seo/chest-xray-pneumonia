#pip install kaggle
import kaggle

# make sure ~/.kaggle/kaggle.json exists in order to use Kaggle API
kaggle.api.authenticate()

dataset_name = "paultimothymooney/chest-xray-pneumonia"
kaggle.api.dataset_download_files(
    dataset_name,
    path = "./data/raw",
    unzip = True
)

#download metadata
kaggle.api.dataset_metadata(dataset_name, path = "./data/raw")
