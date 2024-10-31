# code to retrieve results and analyze from AWS IP of ECs

import numpy as np
import paramiko
import subprocess
import threading
import os
import zipfile
import glob
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import itertools

list_ips = ["52.15.233.96",
            "18.222.218.217",
            "18.222.215.236",
            "18.222.3.15",
            "18.221.61.30",
            "18.217.227.133",
            "18.217.179.7",
            "18.217.56.199",
            "18.190.24.9",
            "18.117.81.242",
            "3.140.244.234",
            "3.128.87.24",
            "3.15.42.232",
            "3.15.33.34",
            "18.223.185.45",
            "3.128.247.188"]


#list_ips = ["52.15.233.96"]
            



output_dir = './results/'
if not os.path.exists(output_dir):
    # Create the folder if it doesn't exist
    os.makedirs(output_dir)
    print(f"Folder created: {output_dir}")
else:
    print(f"Folder already exists: {output_dir}")
    
def check_ssh_connection(hostname, port, username, key_filepath):
    try:
        # Load the private key
        key = paramiko.RSAKey.from_private_key_file(key_filepath)
        
        # Set up SSH client
        ssh = paramiko.SSHClient()
        
        # Automatically add new host keys
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try to connect
        ssh.connect(hostname, port=port, username=username, pkey=key, timeout=5)
        
        # Close the connection if successful
        ssh.close()
        return True
    except paramiko.ssh_exception.SSHException as e:
        print(f"SSH connection failed to {hostname}: {e}")
        return False
    except Exception as e:
        print(f"Failed to connect to {hostname}: {e}")
        return False
    
def zip_on_remote(hostname, port, username, key_filepath, remote_dir, zip_filename):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically add host keys
    try:
        key = paramiko.RSAKey.from_private_key_file(key_filepath)
        client.connect(hostname, port=port, username=username, pkey=key)
        
        # Execute the command
        command = f"zip -r {zip_filename} {remote_dir}"
        stdin, stdout, stderr = client.exec_command(command)

        # Wait for command to complete and avoid blocking
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                print(stdout.channel.recv(1024).decode('utf-8'), end="")  # Read in chunks

            if stderr.channel.recv_stderr_ready():
                print(stderr.channel.recv_stderr(1024).decode('utf-8'), end="")  # Read in chunks

        # Make sure all remaining data in stdout and stderr is printed
        print("Output:")
        for line in stdout:
            print(line.strip())
        print("Errors:")
        for line in stderr:
            print(line.strip())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

def copy_file_from_remote(hostname, username, key_filepath, remote_filepath, local_filepath):
    # Construct the scp command with options to avoid stalling prompts
    scp_command = [
        "scp",
        "-o", "StrictHostKeyChecking=no",           # Automatically accept the host key without prompt
        "-o", "UserKnownHostsFile=/dev/null",       # Avoid writing to known_hosts
        "-i", key_filepath,                         # Path to the SSH private key file
        f"{username}@{hostname}:{remote_filepath}", # Remote file (user@host:path)
        local_filepath                              # Local destination path
    ]

    try:
        # Start the SCP process
        process = subprocess.Popen(scp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Function to read output from a stream without blocking
        def read_stream(stream, label):
            while True:
                line = stream.readline()
                if line:
                    print(f"{label}: {line.strip()}")
                else:
                    break

        # Create threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR"))

        # Start threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for both threads to finish
        stdout_thread.join()
        stderr_thread.join()

        # Wait for the process to complete and check exit status
        process.wait()
        if process.returncode == 0:
            print(f"File copied successfully to {local_filepath}")
        else:
            print(f"An error occurred during SCP transfer. Exit code: {process.returncode}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

def unzip_file(zip_filepath, extract_to_dir):
    # Check if the zip file exists
    if not os.path.exists(zip_filepath):
        print(f"The file {zip_filepath} does not exist.")
        return
    
    # Check if the extract directory exists, if not, create it
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    # Extract the zip file
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
        print(f"Files extracted to {extract_to_dir}")

def read_files_in_folder(folder_path):
    # Dictionary to hold file contents, where keys are filenames and values are arrays
    files_data = {}
    # Array to hold filenames
    filenames = []

    # Get all files in the folder
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        filename = os.path.basename(file_path)  # Extract filename
        
        # Check if "tuning_history" is in the filename
        if "tuning_history" not in filename:
            continue  # Skip this file if it does not contain "tuning_history"

        filenames.append(filename)  # Store filename in list

        # Array to hold lines of current file
        file_content = []

        with open(file_path, 'r') as file:
            # Read each line in the file
            for line in file:
                # Split line by spaces and add to file_content
                file_content.append(line.strip().split())

        # Store content of this file in files_data dictionary
        files_data[filename] = file_content

    return files_data, filenames


def extract_values_from_filenames(folder_path):
    # Regex patterns to capture each float and the string pattern
    pattern_w = r"w_(\d+(\.\d+)?)_(\d+(\.\d+)?)_(\d+(\.\d+)?)_(\d+(\.\d+)?)"
    pattern_a = r"a_(\d+(\.\d+)?)_(\d+(\.\d+)?)"
    pattern_string = r"CACm_(.*?)_AIRBUS"  # Pattern to capture the string between "CACm_" and "_AIRBUS"

    # Lists to store extracted values and filenames
    extracted_data = []
    filenames = []

    # Get all files in the folder
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        filename = os.path.basename(file_path)

        # Only proceed if "tuning_history" is in the filename
        if "tuning_history" not in filename:
            continue  # Skip this file if "tuning_history" is not found

        filenames.append(filename)

        # Extract values for `w_{value1}_{value2}_{value3}_{value4}`
        match_w = re.search(pattern_w, filename)
        if match_w:
            # Convert each matched group to a float
            value1 = float(match_w.group(1))
            value2 = float(match_w.group(3))
            value3 = float(match_w.group(5))
            value4 = float(match_w.group(7))
        else:
            value1 = value2 = value3 = value4 = None  # Default to None if not found

        # Extract values for `a_{a1}_{a2}`
        match_a = re.search(pattern_a, filename)
        if match_a:
            # Convert each matched group to a float
            a1 = float(match_a.group(1))
            a2 = float(match_a.group(3))
        else:
            a1 = a2 = None  # Default to None if not found

        # Extract the string between "CACm_" and "_AIRBUS"
        match_string = re.search(pattern_string, filename)
        extracted_string = match_string.group(1) if match_string else None  # Extract string if found

        # Store extracted values in a dictionary
        extracted_data.append({
            "filename": filename,
            "w_values": (value1, value2, value3, value4),
            "a_values": (a1, a2),
            "extracted_string": extracted_string  # Store the extracted string
        })

    return extracted_data, filenames

# Function to save each cloud of points to a text file
def init_save_cloud_points(string_key, pair_plot, flag):
    filename = f"./results/pareto_{string_key[pair_plot[0]]}_{string_key[pair_plot[1]]}_{flag}.txt"
    new_line = "w1 w2 w3 w4 " + string_key[pair_plot[0]] + " " + string_key[pair_plot[1]] + "\n"
    
    # Read the existing content
    with open(filename, 'r') as f:
        existing_content = f.read()
    
    # Write the new line followed by the existing content
    with open(filename, 'w') as f:
        f.write(new_line + existing_content)
        
    #data = ['w1', 'w2', 'w3', 'w4', string_key[pair_plot[0]], string_key[pair_plot[1]]]
    #with open(filename, 'w') as f:
    #    np.savetxt(f, data, fmt="%s", delimiter=" ")

    
# Function to save each cloud of points to a text file
def save_cloud_points(w1, w2, w3, w4, x, y, string_key, pair_plot, flag):

    filename = f"./results/pareto_{string_key[pair_plot[0]]}_{string_key[pair_plot[1]]}_{flag}.txt"
    # Convert x and y to float and stack as two columns
    data = np.column_stack((w1.astype(float), w2.astype(float), w3.astype(float), w4.astype(float), x.astype(float), y.astype(float)))
    # Append to file in 'a' mode
    with open(filename, 'a') as f:
        np.savetxt(f, data, fmt="%.6f", delimiter=" ")
    
if __name__ == '__main__':
    
    fsetup = "Tune_T=500_R=100"
    
    if False:
        
        for i in range(len(list_ips)):
                
            hostname = list_ips[i]
            port = 22  # Default SSH port
            username = "ubuntu"
            key_filepath = "C:/Users/TimotheeLeleu/Desktop/NTTRI/BMW-AIRBUS_phase2/AWS/QC.pem"  # Path to your .pem key file
            
            # Check if connection can be established
            if not check_ssh_connection(hostname, port, username, key_filepath):
                print(f"Skipping operations for {hostname} due to connection issues.")
                continue  # Skip to the next IP if connection fails
                
            # zip results

            remote_dir = "/home/ubuntu/V22_instance/CAC-DAS_bilevel/Tune_T=500_R=100/"
            zip_filename = f"results_{i}.zip"
            zip_on_remote(hostname, port, username, key_filepath, remote_dir, zip_filename)
         
            # copy file to local machine
        
            remote_filepath = f"results_{i}.zip"
            local_filepath = "./results/"
            copy_file_from_remote(hostname, username, key_filepath, remote_filepath, local_filepath)
     
            # unzip file on local machine 
        
            zip_filepath = f"./results/results_{i}.zip"      # Path to the zip file
            extract_to_dir = f"./results/results_{i}/"  # Directory to extract to
            unzip_file(zip_filepath, extract_to_dir)

    # plot
    if False:
        plt.figure()
        for i in range(len(list_ips)):
        
            # load data
            
            folder_path = f"./results/results_{i}/home/ubuntu/V22_instance/CAC-DAS_bilevel/{fsetup}"  # Replace with your folder path
            data = read_files_in_folder(folder_path)
            if True:
                files_data, filenames = read_files_in_folder(folder_path)
                for filename, data in files_data.items():
                    print(f"\nData for {filename}:")
                    for row in data:
                        print(row)
          
            # parse
            extracted_data, filenames = extract_values_from_filenames(folder_path)
            if True:
                for data in extracted_data:
                    print(f"\nFilename: {data['filename']}")
                    print(f"w_values: {data['w_values']}")
                    print(f"a_values: {data['a_values']}")
                
    
            j=0
            for filename, data in files_data.items():
                fdata = extracted_data[j]
                print(fdata)
                if len(data)>0: #if not empty
                    traj = [[float(row[0]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6])] for row in data]
                    #traj = np.array(traj[1:])
                    traj = np.array(traj)
                    plt.plot(traj[:,0],traj[:,1],'d-',label=fdata['w_values'])
                j+=1
                
        plt.xlabel('n (DAS iteration)')
        plt.ylabel('objective')
        #plt.legend(ncols=3,fontsize=6)
        
        plt.savefig(f"./results/history_{i}.eps", format="eps")
        plt.savefig(f"./results/history_{i}.png", format="png")
        
    
    #Plot function for pareto frontier
    if True:
        
              
        string_key = ['co2','eur','time','ws']
        
        indices_weights = [0,1,2,3]
          
        for u, v in itertools.combinations(indices_weights, 2):
            print(f"Pair: ({u}, {v})")
            
            pair_plot = [u,v]
        
            plt.figure()
            plotted_labels = {}  # Dictionary to track already plotted labels
            colors = list(mcolors.TABLEAU_COLORS.keys())
            color_map = {}
          
            for i in range(len(list_ips)):
            
                # load data
                
                folder_path = f"./results/results_{i}/home/ubuntu/V22_instance/CAC-DAS_bilevel/{fsetup}"  # Replace with your folder path
                data = read_files_in_folder(folder_path)
              
                selected_data = []
                selected_key = []
                selected_flag = []
                
                files_data, filenames = read_files_in_folder(folder_path)
                extracted_data, filenames = extract_values_from_filenames(folder_path)
                
                j=0
                for filename, data in files_data.items():
                    fdata = extracted_data[j]
                    if data:  # Check if data is not empty
                        last_row = data[-1]  # Select the last row
                        selected_values = last_row[3:7]  # Extract elements from index 3 to 6 (inclusive)
                        selected_data.append(selected_values)  # Append to the selected_data array
                        selected_key.append(list(fdata['w_values']))
                        selected_flag.append(fdata['extracted_string'])
                        
                        print(selected_values)
                        
                    j+=1
          
                #select unique flag
                unique_flags = np.unique(selected_flag)
                index_array = [[] for _ in range(len(unique_flags))]
                for i in range(len(unique_flags)):
                    index_array[i] = [j for j in range(len(selected_flag)) if selected_flag[j] == unique_flags[i]]
                print(index_array)
              
                # Assign a color to each unique flag
                for i, flag in enumerate(set(unique_flags)):
                    color_map[flag] = colors[i % len(colors)]  # Cycle through colors if more labels than colors
                
    
                #plot
                
                selected_data = np.array(selected_data)
                selected_key = np.array(selected_key)
                
                for indices, flag in zip(index_array, unique_flags):
                    w1 = selected_key[indices, 0]
                    w2 = selected_key[indices, 1]
                    w3 = selected_key[indices, 2]
                    w4 = selected_key[indices, 3]
                    x = selected_data[indices, pair_plot[0]]
                    y = selected_data[indices, pair_plot[1]]
                    
                    # Save each cloud of points to a text file
                    save_cloud_points(w1, w2, w3, w4, x, y, string_key, pair_plot, flag)
                    
                    for xi, yi in zip(x, y):
                        # Check if the label has already been plotted
                        if flag not in plotted_labels:
                            # Plot with label for legend (only once per unique flag)
                            plt.plot(float(xi), float(yi), 'x', label=flag, color=color_map[flag])
                            plotted_labels[flag] = True  # Mark label as plotted
                        else:
                            # Plot without label to avoid duplicate legend entries
                            plt.plot(float(xi), float(yi), 'x', color=color_map[flag])
                
            plt.xlabel(string_key[pair_plot[0]])
            plt.ylabel(string_key[pair_plot[1]])
            plt.legend()
            
            res_path = './results/'
            filename_base = f"pareto_{string_key[pair_plot[0]]}_{string_key[pair_plot[1]]}"
            plt.savefig(os.path.join(res_path, f"{filename_base}.png"))
            plt.savefig(os.path.join(res_path, f"{filename_base}.eps"))

            init_save_cloud_points(string_key, pair_plot, flag)