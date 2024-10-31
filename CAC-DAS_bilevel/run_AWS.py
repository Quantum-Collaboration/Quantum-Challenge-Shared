import paramiko

import os
import sys
import numpy as np
import glob

import subprocess
import logging

module_dir = os.path.abspath("./../code")
#module_dir = os.path.abspath("./../optimize")
sys.path.append(module_dir)

from cache import weight_generator


def execute_remote_command_single_ip(hostname, username, key_filepath, remote_dir, command, background_command=None):
    print(f"\nConnecting to {hostname}...")

    # Initialize the SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Load the SSH key
        key = paramiko.RSAKey.from_private_key_file(key_filepath)

        # Connect to the remote server
        client.connect(hostname, port=22, username=username, pkey=key)

        # Execute the background command with nohup and &, combined with cd
        # Redirect stdout and stderr to ensure no output is expected on the channel
        full_command = f"cd {remote_dir} && nohup {command} > /dev/null 2>&1 &"
        stdin, stdout, stderr = client.exec_command(full_command)

        print(f"Started background command on {hostname}: {command}")

        # If another command is provided, run it immediately after
        if background_command:
            full_command = f"cd {remote_dir} && {background_command}"
            stdin, stdout, stderr = client.exec_command(full_command)

            # Read and print output of the second command
            print(f"Output from second command on {hostname}:")
            for line in stdout:
                print(line.strip())
            print(f"Errors from second command on {hostname}:")
            for line in stderr:
                print(line.strip())

    except Exception as e:
        print(f"An error occurred on {hostname}: {e}")

    finally:
        # Close the SSH connection for this IP
        client.close()


def update_code(hostname, username, key_filepath):
    # Define remote directory and find .py files locally
    remote_filepath = '/home/ubuntu/V22_instance/CAC-DAS_bilevel/'
    local_files = glob.glob('./*.py')  # Expands wildcard to list of .py files

    if not local_files:
        print("No .py files found in the current directory.")
        return

    for local_filepath in local_files:
        print(f"Starting transfer of {local_filepath} to {hostname}...")

        scp_command = [
            "scp",
            "-i", key_filepath,                        # Path to .pem key file
            "-o", "StrictHostKeyChecking=no",          # Skip SSH key verification prompts
            local_filepath,                            # Local .py file
            f"{username}@{hostname}:{remote_filepath}" # Remote directory path
        ]

        try:
            # Run the scp command with real-time output
            result = subprocess.run(scp_command, check=True, text=True)

            print(f"File {local_filepath} copied successfully to {remote_filepath} on {hostname}")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred with {local_filepath}: {e.stderr}")

def test_ssh_connection(hostname, username, key_filepath):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically add unknown host keys

    try:
        # Load the SSH key
        key = paramiko.RSAKey.from_private_key_file(key_filepath)

        # Connect to the remote server
        client.connect(hostname, port=22, username=username, pkey=key)
        print(f"Successfully connected to {hostname}")

    except Exception as e:
        print(f"An error occurred on {hostname}: {e}")

    finally:
        client.close()


# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Example test function with logging
def test_ssh_connection_with_logging(hostname, username, key_filepath):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        key = paramiko.RSAKey.from_private_key_file(key_filepath)
        client.connect(hostname, port=22, username=username, pkey=key)
        print(f"Successfully connected to {hostname}")

    except Exception as e:
        print(f"An error occurred on {hostname}: {e}")

    finally:
        client.close()



if __name__ == '__main__':
    
    list_ips = ["52.15.233.96",
                "18.224.172.199",
                "18.223.247.165",
                "18.222.218.217",
                "18.222.215.236",
                "18.222.3.15",
                "18.221.61.30",
                "18.217.227.133",
                "18.217.179.7",
                "18.217.56.199",
                "18.190.24.9",
                "18.117.81.242",
                "13.59.189.95",
                "13.59.39.107",
                "3.140.244.234",
                "3.138.169.179",
                "3.132.216.154",
                "3.128.87.24",
                "3.15.42.232",
                "3.15.33.34",
                "18.223.185.45",
                "3.128.247.188"]
    
    username = "ubuntu"
    key_filepath = "C:/Users/TimotheeLeleu/Desktop/NTTRI/BMW-AIRBUS_phase2/AWS/QC.pem"
    remote_dir = "/home/ubuntu/V22_instance/CAC-DAS_bilevel/"
    
    
    # Create list of commands

    N = 10
    digits = 3

    gen = weight_generator(N,digits)

    # Accessing the yielded list
    list_commands = []
    for weights in gen:
        print(weights)
        
        #TIER1
        list_commands.append(f'python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning')
        list_commands.append(f'python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning -run_BP -run_QAOA')
        
        #TIER2
        list_commands.append(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning -use_model &\n')
        #list_commands.append(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel &\n')
        
        
        #list_commands.append(f'nohup python3 main_AIRBUS.py {weights[0]} {weights[1]} {weights[2]} {weights[3]} 4 5 -reloadModel -online_tuning -use_model -run_BP -run_QAOA &\n')


    Num_commands = len(list_commands)
    per_machine = 31
    Num_machines = np.ceil(Num_commands/per_machine)
    Num_CPU = Num_machines * 36
    print(f"Num_machines = {Num_machines}")
    print(f"Num_CPU = {Num_CPU}")
    
    price_hour = 1.5
    time = 24
    total_price = Num_machines * time * price_hour
    print(f"Total price = {total_price}")
    
    DORUN = True
    
    #run all
    if True:
        
        Num_machines1 = int(np.floor(Num_commands/per_machine))
        rem = int(Num_commands - Num_machines1*per_machine)
        print(f"Running total { Num_machines1*per_machine + rem} out of {Num_commands}")
        
        list_commands_machine1 = np.reshape(list_commands[:int(Num_machines1*per_machine)],[Num_machines1,per_machine])
        list_commands_machine2 = list_commands[-rem:]
        
        count = 0
        countm = 0
        for ip, commands in zip (list_ips,list_commands_machine1):
            #print(ip,commands)
            if DORUN:
                update_code(ip, username, key_filepath)
            print("--------- Code copied ------------")
            countm += 1
            for command in commands:
                #test_ssh_connection(ip, username, key_filepath)
                #test_ssh_connection_with_logging(ip, username, key_filepath)
                
                if DORUN:
                    execute_remote_command_single_ip(ip, username, key_filepath, remote_dir, command.strip())
                count += 1
                #ERROR
                
        ip = list_ips[countm]
        commands = list_commands_machine2
        update_code(ip, username, key_filepath)
        countm += 1
        for command in commands:
            if DORUN:
                execute_remote_command_single_ip(ip, username, key_filepath, remote_dir, command.strip())
            count += 1
            
        print(f"ran {countm} out of {Num_machines} machines")
        print(f"ran {count} out of {Num_commands} commands")
        
