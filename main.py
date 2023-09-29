import os, pretty_errors, argparse, subprocess
from model import test_model, load_model

os.environ['KAGGLE_CONFIG_DIR']="./kaggle"

DESTINATION_DIRECTORY = './yolov5'
RUNS_DIRECTORY = './runs'

IMG_SIZE = 640
BATCH_SIZE = 16
NUM_EPOCHS = 3
DEFAULT_MODEL = 'yolov5s'
TESTING_SOURCE = "./"
def install_reqs():
    pip_install_command = ['pip', 'install', '-r', f'requirements.txt']
    try:
        subprocess.check_output(pip_install_command, stderr=subprocess.STDOUT, shell=True)
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e.output.decode('utf-8')}")

def download_dataset():
    if(not os.path.exists("./kaggle")):
        os.makedirs("./kaggle")
        print("Kaggle datoteka kreirana")

    if(not os.path.isfile("./kaggle/kaggle.json")):
        kaggle_username = input("Unesite svoje Kaggle korisnicko ime: ")
        kaggle_api_key = input("Unesite svoj Kaggle API key: ")

        import json

        with open("./kaggle/kaggle.json", mode="x") as file:
            file.write(json.dumps(
                {"username":f"{kaggle_username}","key":f"{kaggle_api_key}"}
            ))
        print("Kaggle.json created at ./kaggle")
    
    if(not os.path.exists("./dataset")):
        os.makedirs("./dataset")
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset='yusufberksardoan/traffic-detection-project', path='./dataset', unzip=True)
            print("Dataset downloaded!")
        except Exception as error:
            print(f"Greska prilikom preuzimanja:\t{error}")
            os.remove("./kaggle/kaggle.json")
            print("kaggle.json uklonjen")
            os.rmdir("./dataset")
            print("Prazni direktorijum dataset uklonjen.")
           


def clone_yolov5_training_repo():
    # Specify the directory where you want to clone the repository
    if(not os.path.exists(DESTINATION_DIRECTORY)):

        # Replace with the URL of the Git repository you want to clone
        repository_url = 'https://github.com/ultralytics/yolov5'

        # Construct the Git command to clone the repository
        git_command = ['git', 'clone', repository_url, DESTINATION_DIRECTORY]

        # Execute the Git command
        try:
            subprocess.check_output(git_command, stderr=subprocess.STDOUT, shell=True)
            print(f"Repository cloned successfully to '{DESTINATION_DIRECTORY}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.output.decode('utf-8')}")
            
    pip_install_command = ['pip', 'install', '-r', f'{DESTINATION_DIRECTORY}/requirements.txt']
    try:
        subprocess.check_output(pip_install_command, stderr=subprocess.STDOUT, shell=True)
        print("Yolov5 repo requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e.output.decode('utf-8')}")

    if(not os.path.exists(RUNS_DIRECTORY)):
        os.mkdir(RUNS_DIRECTORY)
        print("runs direktorijum napravljen.")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL, help='Ime yolov5 modela.')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='Velicina slike (hxw)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size za trening')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Broj epoha za trening')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument("-train", "--train", action="store_true", help="If you want to train the model")
    parser.add_argument("-test", "--test", action="store_true", help="If you want to perform inference on a specific file to be uploaded")
    parser.add_argument("-val", "--validate", action="store_true", help="If you want to perform validation")
    parser.add_argument("-half", "--half", action="store_true", help="If you want to use half(FP16) precision")
    parser.add_argument("--source", type=str, default=TESTING_SOURCE, help="Putanja ka source-u za testiranje")
    parser.add_argument("--weights", type=str, default=f"./{DEFAULT_MODEL}.pt", help="If you want to perform validation")
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping broj koraka za zaustavljanje')

    args = parser.parse_args()
    return args

def main(args):
    install_reqs()
    download_dataset()
    clone_yolov5_training_repo()
    # The following pre-processing was applied to each image:
    # * Auto-orientation of pixel data (with EXIF-orientation stripping)
    # * Resize to 640x640 (Stretch)
    # The following augmentation was applied to create 3 versions of each source image:
    # * 50% probability of horizontal flip
    # * Random brigthness adjustment of between -25 and +25 percent
    # * Salt and pepper noise was applied to 2 percent of pixels

    # model = load_model(args.model_name)
    image_size = args.img_size 
    batch_size = args.batch_size
    epochs = args.epochs
    weights = args.weights
    is_half = args.half
    source = args.source
    freeze = args.freeze
    optimizer = args.optimizer
    patience = args.patience
    resume = args.resume
    model_name = args.model_name
    model_train_command = [
        'python', f'./{DESTINATION_DIRECTORY}/train.py', 
        '--img', f'{image_size}', 
        '--batch', f'{batch_size}', 
        '--epochs', f'{epochs}', 
        '--data', 'dataset.yaml', 
        '--device', '0', 
        '--project', './runs/train', 
        '--weights', f'{model_name}.pt',
        '--freze', f'{freeze}',
        '--resume', f'{resume}'
        '--optimizer', f'{optimizer}',
        '--patience', f'{patience}'
    ]

    model_validation_command = [
        'python', f'./{DESTINATION_DIRECTORY}/val.py', 
        '--imgsz', f'{image_size}', 
        '--batch-size', f'{batch_size}', 
        # '--epochs', f'{epochs}', 
        '--data', 'dataset.yaml', 
        '--device', '0', 
        '--project', './runs/valid', 
        '--weights', weights, 
        # '--half' if is_half else ''
    ]
    
    model_testing_command = [
        'python', f'./{DESTINATION_DIRECTORY}/detect.py', 
        '--weights', weights ,
        '--source', source,
        # '--data', 'dataset.yaml',  # NOTE opcionalno za testiranje
        '--imgsz', f'{image_size}', 
        '--device', '0',
        '--project', './runs/test', 
    ]

    if args.train:
        if os.path.exists("./runs/train/exp"):
            from shutil import rmtree
            rmtree("./runs/train/exp")


        print("Training starting")
        print("-"*10)
        training = subprocess.Popen(model_train_command, stderr=subprocess.STDOUT, shell=True)
        try:
            training.wait()
        except KeyboardInterrupt:
            training.terminate()
            print("Trening prekinut")
        except subprocess.CalledProcessError as e:
            print(f"Error during training:\n{e.output.decode('utf-8')}")

        print("-"*10)
        print("Training finished.")
    elif args.validate:
        print("Validation starting")
        print("-"*10)
        validation = subprocess.Popen(model_validation_command, stderr=subprocess.STDOUT, shell=True)
        try:
            validation.wait()
        except KeyboardInterrupt:
            validation.terminate()
            print("validacija prekinuta")
        except subprocess.CalledProcessError as e:
            print(f"Error during validation:\n{e.output.decode('utf-8')}")

        print("-"*10)
        print("Validation finished.")
    elif args.test:
        print("Testing starting")
        print("-"*10)
        testing = subprocess.Popen(model_testing_command, stderr=subprocess.STDOUT, shell=True)
        try:
            testing.wait()
        except KeyboardInterrupt:
            testing.terminate()
            print("testiranje prekinuta")
        except subprocess.CalledProcessError as e:
            print(f"Error during testing:\n{e.output.decode('utf-8')}")

        print("-"*10)
        print("Testing finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)