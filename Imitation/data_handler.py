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

def read_data(filename):

    filename = str(mod_path) + "/" + filename
    x_values = []
    y_values = []

    with open(filename) as csvfile:

        reader = csv.reader(csvfile)
        for row in reader:
            x_values.append(eval(row[0]))
            y_values.append(int(row[1]))

    return x_values, y_values