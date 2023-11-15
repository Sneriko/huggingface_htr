from .HTRDataset import HTRDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

class Utils():
    def __init__(self):
        pass


    def create_datasets(basepaths: list, gt_paths: list, train_eval_split: float, processor):
        datasets_train = list()
        datasets_test = list()

        for basepath, gt in zip(basepaths, gt_paths):
            df = pd.read_json(gt, lines=True)
            df.rename(columns={0: "filename", 1: "text"}, inplace=True)
            # df = df[0:2100]

            train_df, test_df = train_test_split(df, test_size=train_eval_split)
            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)

            train_dataset = HTRDataset(root_dir=basepath, df=train_df, processor=processor)
            eval_dataset = HTRDataset(root_dir=basepath, df=test_df, processor=processor)

            datasets_train.append(train_dataset)
            datasets_test.append(eval_dataset)

        len_of_datasets = sum(len(sublist) for sublist in datasets_train)

        train_concat_dataset = ConcatDataset(datasets_train)
        test_concat_dataset = ConcatDataset(datasets_test)

        return train_concat_dataset, test_concat_dataset