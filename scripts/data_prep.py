import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


class Clean_df:
    def get_numerical_columns(self, df: pd.DataFrame) -> list:
        """
        Returns numerical column names
        """
        return df.select_dtypes(include='number').columns

    def drop_null_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if there is a null entry in the dataset and removes them
        """
        df.dropna(subset=df.columns, axis=0, inplace=True)
        return df

    def label_encode(self, df: pd.DataFrame, col_names: list) -> pd.DataFrame:
        """ Performs Label encoding of the given columns

        Parameters
        ------------
        df: Pandas DataFrame: dataframe to be computed
        Columns: list of columns
        Returns
        ------------
        The method returns a dataframe with label encoded categorical features
        """

        le = LabelEncoder()
        for col in col_names:
            df[col+'_encoded'] = le.fit_transform(df[col])

        df.drop(columns=col_names, axis=1,inplace=True)
        return df

    def one_hot_encode(self, df: pd.DataFrame, col_names: list) -> pd.DataFrame:
        """ Performs One hot encoding of the given columns

        Parameters
        ------------
        df: Pandas DataFrame: dataframe to be computed
        Columns: list of columns
        Returns
        ------------
        The method returns a dataframe with One-hot encoded categorical features
        """
        # ohe = OneHotEncoder(handle_unknown='ignore')

        return pd.get_dummies(df, columns=col_names)

    def minmax_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns dataframe with minmax scaled columns
        """
        scaller = MinMaxScaler()
        res = pd.DataFrame(
            scaller.fit_transform(
                df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df)
        )
        return res

    def normalizer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns dataframe with normalized columns
        """
        nrm = Normalizer()
        res = pd.DataFrame(
            nrm.fit_transform(
                df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df)
        )
        return res

    def map_brands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The following maps shows the device make column into known and unknowncolumns
        """
        known_brands = ['samsung', 'htc', 'nokia',
                        'moto', 'lg', 'oneplus',
                        'iphone', 'xiaomi', 'huawei',
                        'pixel']
        makers = ["Unknown"]*df.shape[0]
        for idx, make in enumerate(df['device_make'].values):
            for brand in known_brands:
                if brand in make.lower():
                    makers[idx] = brand
                    break
        df['device_make'] = makers
        return df

    def collect_reponse(self, df: pd.DataFrame):
        """
        The following method removes the yes and no columns by merging their result into 'response'
        column
        """
        df['response'] = df['yes']
        df.drop(columns=['yes', 'no'], inplace=True)
        return df

    def drop_unwanted_cols(self, df: pd.DataFrame, col_names: list = None) -> pd.DataFrame:
        """
        Drops columns which are not necessary for model training
        """
        if not col_names:
            col_names = ['Unnamed: 0', 'auction_id',
                         'date',  'yes', 'no','experiment_encoded']
        for col in col_names:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        return df

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        performs a pipiline of cleaning methods in the given dataframe
        """
        df = self.drop_null_entries(df)
        df = self.map_brands(df)
        df = self.label_encode(df, col_names=['experiment'])
        df = self.one_hot_encode(df, col_names=['device_make'])
        df_temp = df[['yes','experiment_encoded']]
        df = self.drop_unwanted_cols(df)
        
        df = self.minmax_scaling(df)
        df = self.normalizer(df)
        df = df.join(df_temp)
        df.reset_index(drop=True, inplace=True)
        return df
    
