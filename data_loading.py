#### IMPORTS AND PATHS
import pandas as pd
import os
from shutil import move
import itertools
from pathlib import Path
data_dir = '../data/hands/Hands/Hands'

handsinfo = pd.read_csv("../data/hands/HandInfo.csv")


### METHODS
def train_test_split_for_img_data(df, train_frac=0.6, val_frac=0.2):
    test_frac = val_frac/(1-train_frac)
    handstrain = pd.DataFrame(df).sample(frac=train_frac, random_state=25)
    handsval_interim = pd.DataFrame(df).drop(handstrain.index)
    handsval = pd.DataFrame(handsval_interim).sample(frac=test_frac, random_state=25)
    handstest = pd.DataFrame(handsval_interim).drop(handsval.index)

    handstrain = pd.Series(handstrain[0]).tolist()
    handsval = pd.Series(handsval[0]).tolist()
    handstest = pd.Series(handstest[0]).tolist()
    
    return handstrain, handsval, handstest

def split_img_data(df, target_col="irregularities", image_col="imageName", train_frac=0.6, val_frac=0.2):

    handsirreg = df.loc[lambda x:x[target_col] == 1, image_col].copy().tolist()
    handsreg = df.loc[lambda x:x[target_col] == 0, image_col].copy().tolist()

    handsregtrain, handsregval, handsregtest = train_test_split_for_img_data(handsreg, train_frac=train_frac, val_frac=val_frac)
    handsirregtrain, handsirregval, handsirregtest = train_test_split_for_img_data(handsirreg, train_frac=train_frac, val_frac=val_frac)
    
    return handsregtrain, handsregval, handsregtest, handsirregtrain, handsirregval, handsirregtest

def join_tuple_string(strings_tuple) -> str:
    return '/'.join(strings_tuple) + '/'


####### EXECUTION

#### CREATE SUBFOLDERS
subfolders1 = ["reg", "irreg"]
subfolders2 = ["train", "val", "test"]

subfolders = list(itertools.product(subfolders1, subfolders2))
# joining all the tuples
subfolders = list(map(join_tuple_string, subfolders))

#### SPLIT DATA
handsregtrain, handsregval, handsregtest, handsirregtrain, handsirregval, handsirregtest = split_img_data(handsinfo)


#### SEPARATE DATA INTO FOLDERS
for nu, subfolder in enumerate(subfolders):
    print(subfolder)
    print(f"creating {data_dir}/{subfolder}")
    new_path = f"{data_dir}/{subfolder}"
    print(new_path)
    if "train" in new_path:
        if "irreg" in new_path:
            df = handsirregtrain
            print("in train irreg")
        else:
            df = handsregtrain
            print("in train reg")
    elif "val" in new_path:
        if "irreg" in new_path:
            df = handsirregval
            print("in val irreg")
        else:
            df = handsregval
            print("in val reg")

    elif "test" in new_path:
        if "irreg" in new_path:
            df = handsirregtest
            print("in test irreg")

        else:
            df = handsregtest
            print("in test reg")
    Path(new_path).mkdir(parents=True, exist_ok=True)
    if os.path.exists(new_path):
        pass
    else:
        if "val" in new_path:
            if os.path.exists(f'{data_dir.replace("val", "train")}/{img}'):
                [move(f'{data_dir.replace("val", "train")}/{img}', f"{new_path}/{img}") for img in df]
            elif os.path.exists(f'{data_dir.replace("val", "test")}/{img}'):
                [move(f'{data_dir.replace("val", "test")}/{img}', f"{new_path}/{img}") for img in df]
            else:
                [move(f"{data_dir.split('11')[0]}/{img}", f"{new_path}/{img}") for img in df]
        else:        
            [move(f"{data_dir.split('11')[0]}/{img}", f"{new_path}/{img}") for img in df]