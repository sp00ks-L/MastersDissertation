# Title      :	main.py
# Objective  :
# Created by :	Luke
# Created on :	Tue 24/08/21 14:20

from pathlib import Path

from data_gen.chromo_sampler import chromo_sampler, clear_folders
from data_gen.data_labelling import generate_labels
from models.final_model import create_model, load_and_split, train_and_save

cwd = Path(__file__).parent.absolute()
data_path = cwd.joinpath("data")


def check_user_input(user_input):
    if user_input.lower() in ['yes', 'y', 'ye']:
        return True
    elif user_input.lower() in ['no', 'n']:
        return False
    else:
        raise ValueError("Enter either yes | no")

# 1 :: Label the data


def labels_exist():

    pass


def label_data():
    print("\n---------------------------------\nMake sure the raw csv files are in the raw_data directory before you continue\n---------------------------------\n")
    usr_in = input("Produce labels for all 3 chromosomes?   : Yes|No\n")
    if check_user_input(usr_in):
        generate_labels(data_path, chromosome_number=1)
        generate_labels(data_path, chromosome_number=2)
        generate_labels(data_path, chromosome_number=3)
    elif not check_user_input(usr_in):
        chr_num = int(
            input("Which chromosome would you like to produce data for?     : 1|2|3\n"))
        generate_labels(data_path, chromosome_number=chr_num)

# 2 :: Sample the data


def make_dirs(data_dir=data_path):
    """[Makes the necessary folders for data generation in the cwd]

    Args:
        data_dir ([Pathlib.Path], optional): [File path that points towards the data folder]. 
        Defaults to data_path.
    """
    model_data_path = data_dir.joinpath('model_data')
    parents = ["Train", "Test", "Validate"]
    children = ["Binding", "Non-binding"]
    for parent in parents:
        for child in children:
            try:
                Path.mkdir(model_data_path.joinpath(
                    parent, child), parents=True)
            except FileExistsError:
                pass
    try:
        Path.mkdir(data_dir.joinpath("raw_data"))
    except FileExistsError:
        pass

    print("Data Folders Created.")


def sample_data():
    usr_in = input(
        "Do you want to clear the current Train, Test and Validate Folders?     : Yes|No\n")
    if check_user_input(usr_in):
        # TODO Probably add a further check here to prevent accidental mis-types / clicks
        # TODO Maybe check if folders are empty to skip this check
        clear_folders(data_path.joinpath("model_data"))
    usr_sample_size = int(input("Please input a sample size: "))
    usr_window_size = int(
        input("Please input a window_size (default = 5000): "))
    usr_chromo_check = input(
        "Would you like to sample all 3 chromosomes?   : Yes|No")
    if check_user_input(usr_chromo_check):
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=1)
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=2)
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=3)
    elif not check_user_input(usr_chromo_check):
        usr_chromo_number = int(
            input("Which chromosome would you like to sample?   : 1|2|3"))
        # TODO implement the window_size check
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=usr_chromo_number)

    else:
        # do all 3 by default
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=1)
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=2)
        chromo_sampler(data_path, usr_sample_size,
                       window_size=usr_window_size, chromosome_number=3)


# 3 :: Instantiate the model and train
def train_model():
    # TODO If model exists maybe ask if overwrite is okay
    model_data_path = data_path.joinpath("model_data")
    train_dir = model_data_path.joinpath("Train")
    test_dir = model_data_path.joinpath("Test")
    validate_dir = model_data_path.joinpath("Validate")
    train_x, train_y, test_x, test_y, val_x, val_y = load_and_split(
        train_dir, test_dir, validate_dir)
    cnn_model = create_model()
    train_and_save(cnn_model, train_x, train_y, val_x, val_y)
    print("\nTraining Completed.\n")


def main():
    # make_dirs()
    # label_data()
    # sample_data()
    train_model()


if __name__ == "__main__":
    main()
