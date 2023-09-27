import torch, os, pretty_errors

os.environ['KAGGLE_CONFIG_DIR']="./kaggle"

def download_dataset():
    if(not os.path.exists("./dataset")):
       os.makedirs("./dataset")
       import kaggle
       kaggle.api.authenticate()
       kaggle.api.dataset_download_files(dataset='yusufberksardoan/traffic-detection-project', path='./dataset', unzip=True)
       print("Dataset downloaded!")

def main():
    download_dataset()

if __name__ == "__main__":
    main()