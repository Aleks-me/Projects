import librosa
import numpy as np
import pandas as pd


class DfCreator:
    def __init__(self,
                 sound_data: str,
                 sample_rate: int,
                 n_features: int):
        self.s_data = sound_data
        self.s_rate = sample_rate
        self.n_f = n_features

    def __repr__(self):
        """Class for reading .wav files and dataframe (df) creation.

        Methods:
            file_reader: read wav to MFCC vectors.
            make_db: create df from the vectors.
            db_check_for_nulls: sanity check for not nulls in df.
            make_test_db: make test df.
            make_train_db: make train df.
        """

    def file_reader(self, audio_list: str) -> np.array:
        """Function that reads wav data to vectors.

        Input:
            audio_list (str): file name with audio list.

        Returns:
            np.array: matrix with the number of rows == number of audio
                      files and n_features == number of columns.
        """
        features = np.array(())

        try:
            with open(audio_list, "r") as file:
                readed = file.read().splitlines()
        except IOError as ioe:
            print(ioe)
        else:
            for r_file in readed:
                encoded, sr = librosa.load(self.s_data + "/" + r_file,
                                           sr=self.s_rate)
                matrix = librosa.feature.mfcc(encoded, sr=self.s_rate,
                                              n_mfcc=self.n_f).T
                vector = np.mean(matrix, axis=0)
                if features.size == 0:
                    features = vector
                else:
                    features = np.vstack((features, vector))
        return features

    @staticmethod
    def make_db(features_list: list) -> pd.DataFrame:
        """Function to create df from features matrices.

        Input:
            features_list (list): list of features matrices.

        Returns:
            pd.DataFrame: united dataframe.
        """

        concat_list = []
        for i, elem in enumerate(features_list):
            db = pd.DataFrame(elem)
            # We know what .wav files were recorded by male or female
            # speaker, so let's make column for supervised learning:
            db['target'] = i + 1  # 1 for female speaker, 2 for male
            concat_list.append(db)

        df_full = pd.concat(concat_list, ignore_index=True)
        return df_full

    @staticmethod
    def db_check_for_nulls(df: pd.DataFrame) -> bool:
        """Sanity check for none nulls in dataframe (df).

        Input:
            df (pd.DataFrame): df.

        Returns:
            bool: True if there is at least one null value,
                  otherwise - False.
        """
        has_nulls = df.isnull().values.any()
        return has_nulls

    @staticmethod
    def make_test_db(dataframe: pd.DataFrame,
                     sample: int) -> pd.DataFrame:
        """Make test dataframe (df).

        Input:
            dataframe (pd.DataFrame): df.
            sample (int): number of rows for random sampling.

        Returns:
            test_df (pd.DataFrame): df of sample rows.
        """
        test_df = dataframe.sample(n=sample).reset_index(drop=True)
        return test_df

    @staticmethod
    def make_train_db(dataframe: pd.DataFrame,
                      test_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Make train dataframe (df).

        Input:
            dataframe (pd.DataFrame): df.
            test_dataframe (pd.DataFrame): test df.

        Returns:
            train_df (pd.DataFrame): train_df = full_df - test_df.
        """
        train_df = dataframe.drop(index=test_dataframe.index)
        train_df.reset_index(drop=True, inplace=True)
        return train_df
