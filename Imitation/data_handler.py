from pathlib import Path
mod_path = Path(__file__).parent
import csv
import time

start_time = str(time.time()).split(".")[0][-5:]
filename = str(mod_path) + "/data/" + "data_" + start_time + ".csv" 

def write_data(state, action):

    with open(filename, "a", newline='') as csvfile:
        
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([state, action])

def read_data(filename):

    filename = str(mod_path) + "/data/" + filename
    x_values = []
    y_values = []

    with open(filename) as csvfile:

        reader = csv.reader(csvfile)
        for row in reader:
            x_values.append(eval(row[0]))
            temp = [0]*5
            temp[int(row[1])] = 1
            y_values.append(temp)

    return x_values, y_values