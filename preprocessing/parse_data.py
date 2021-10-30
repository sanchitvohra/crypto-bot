import pandas as pd
import numpy as np
import os, sys

data_files_path = sys.argv[1]

def combine_files():
    symbs = ["BCH", "BTC", "ETH", "LTC", "XRP"]
    years = [2017, 2018, 2019, 2020, 2021]

    for symb in symbs:
        frames = []
        for year in years:
            filename = "Bitstamp_" + symb + "USD_" + str(year) + "_minute.csv"
            frames.append(pd.read_csv(os.path.join(data_files_path, filename), low_memory=False))
        combined = pd.concat(frames)
        print("Writing combined " + symb)
        combined.to_csv(os.path.join(data_files_path, "Bitstamp_" + symb + "USD.csv"), index=False)

combine_files()
