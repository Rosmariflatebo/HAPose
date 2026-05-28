# If you will run the program with the actual vest set the demo_game to False 
import pandas as pd
from pathlib import Path
from user_interface import start_UI

demo_game = True 

demo_data_over_time = {"Date": ["19.02", "23.02", "28.02", "05.03", "12.03", "19.03", "26.03"], 
                       "Performance value": [3.4, 3.9, 4.2, 3.5, 4, 4.2, 4.5]} #demo data for the performance value over time
demo_data_today = [4, 4, 3, 4, 3, 4, 4, 5] #demo data for 2 hour of wearing the vest

file = "vest stats over time.csv"

def start_data():
    data = {"Date": [], "Performance value": []}
    return pd.DataFrame(data)

def load_start_data():
    if Path(file).exists():
        return pd.read_csv(file)
    else:
        df = start_data()
        df.to_csv(file, index=False)
        return df


if __name__ == "__main__":
    if demo_game == False:
        data = load_start_data()
        data_today = []

    if demo_game == True:
        data = demo_data_over_time
        data_today = demo_data_today
        start_UI(data, data_today)
    



    pass