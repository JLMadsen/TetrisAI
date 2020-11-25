import os
import threading

print_lock = threading.Lock()

def run_file(filename):
    
    with print_lock:
        print('Start', filename)
        
    os.system('python '+ filename)

files = ['main_dqn.py', 'main_imitation.py', 'main_natselect.py']
threads = []

for file in files:
    threads.append( threading.Thread(target=run_file, args=(file,)) )
    threads[-1].start()
    
for thread in threads:
    thread.join()
    