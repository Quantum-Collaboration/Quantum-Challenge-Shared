import os
import sys
import numpy as np

module_dir = os.path.abspath("./../code")
#module_dir = os.path.abspath("./../optimize")
sys.path.append(module_dir)

from cache import weight_generator


N = 5
digits = 3

gen = weight_generator(N,digits)

# Accessing the yielded list
with open("tasks.bat", "w") as file:
    for weights in gen:
        print(weights)  # Output: [1, 2, 3, 4]
        #file.write(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel &\n')
        file.write(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning &\n')
        #file.write(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning -use_model &\n')
        #file.write(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning -run_BP -run_QAOA &\n')
        #file.write(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning -use_model -run_BP -run_QAOA &\n')

    
python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel
python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning
python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning -use_model
python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning -run_BP -run_QAOA
#python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning -use_model -run_BP -run_QAOA

