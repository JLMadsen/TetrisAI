from pathlib import Path
mod_path = Path(__file__).parent
import csv
import time

start_time = str(time.time()).split(".")[0][-5:]
filename = str(mod_path) + "/" + "data_" + start_time + ".csv" 

def write_data(state, action):

    with open(filename, "a", newline='') as csvfile:
        
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([state, action])