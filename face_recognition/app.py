import sys
import os
from pathlib import Path
from dataloader import get_train_data, get_test_data, split, processing_data_augmentation, create_test_environment, remove_files_in_folder
from training import training
from testing import testing

root_path = Path(__file__).parent.parent

# Add paths for data


POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

TEST_INPUT_PATH = os.path.join("data", "test", "input")
TEST_POS_PATH = os.path.join("data", "test", "positive")

# make directories

if sys.argv[1] == "train":
    try:
        if len(sys.argv) >= 3:
            try:
                os.makedirs(POS_PATH)
            except:
                print("you have already positive folder!", flush=True)

            try:
                os.makedirs(ANC_PATH)
            except:
                print("you have already anchor (input) folder!", flush=True)

            DATA_POOL_ADDRESS = str(sys.argv[2])
            print("Creating Folders. [1/4]", flush=True)
            # split data
            epoch = int(sys.argv[3]) if len(sys.argv) == 4 else 5
            split(root_path, DATA_POOL_ADDRESS, POS_PATH, ANC_PATH)
            processing_data_augmentation("positive", POS_PATH, ANC_PATH)
            processing_data_augmentation("anchor", POS_PATH, ANC_PATH)
            train_loader, val_loader = get_train_data(anc_path=ANC_PATH, neg_path=NEG_PATH, pos_path=POS_PATH)
            training(train_loader, val_loader, epoch, save_model=True)

        else:
            print("Invalid Input", flush=True)
    except:
        remove_files_in_folder(POS_PATH)
        remove_files_in_folder(ANC_PATH)



elif sys.argv[1] == "test":
    try:
        if len(sys.argv) >= 3:
            create_test_environment(root_path, sys.argv[2])
            name = sys.argv[3] if len(sys.argv) == 4 else 'UNKNOWN'
            input_loader, file_name = get_test_data(TEST_INPUT_PATH, TEST_POS_PATH)
            testing(input_loader, file_name, name)
            remove_files_in_folder(TEST_POS_PATH)
        else:
            print("Invalid Input", flush=True)
    except:
        remove_files_in_folder(TEST_POS_PATH)


else:
    print("Invalid Input", flush=True)
