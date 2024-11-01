import os
import zipfile
import glob
import re

import paramiko

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
            "3.15.33.34"]

            
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
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try to connect
        ssh.connect(hostname, port=port, username=username, pkey=key, timeout=5)
        
        # Close the connection if successful
        ssh.close()
        return True
    except Exception as e:
        print(f"Failed to connect to {hostname}: {e}")
        return False
    
def delete_in_folder(hostname, port, username, key_filepath, remote_dir):
    try:
        # Load the private key
        key = paramiko.RSAKey.from_private_key_file(key_filepath)
        
        # Set up SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the remote host
        ssh.connect(hostname, port=port, username=username, pkey=key)
        
        # Command to delete all files in the specified directory
        delete_command = f"rm -rf {remote_dir}/*"
        
        # Execute the delete command
        stdin, stdout, stderr = ssh.exec_command(delete_command)
        error = stderr.read().decode()
        
        if error:
            print(f"Error deleting files in {remote_dir} on {hostname}: {error}")
        else:
            print(f"Successfully deleted all files in {remote_dir} on {hostname}")
        
        # Close the SSH connection
        ssh.close()
    except Exception as e:
        print(f"Failed to delete files in {remote_dir} on {hostname}: {e}")


if __name__ == '__main__':
    
    fsetup = "Tune_T=500_R=100"
    
    for i in range(len(list_ips)):
            
        hostname = list_ips[i]
        port = 22  # Default SSH port
        username = "ubuntu"
        key_filepath = "C:/Users/TimotheeLeleu/Desktop/NTTRI/BMW-AIRBUS_phase2/AWS/QC.pem"  # Path to your .pem key file
        
        # Check if connection can be established
        if not check_ssh_connection(hostname, port, username, key_filepath):
            print(f"Skipping operations for {hostname} due to connection issues.")
            continue  # Skip to the next IP if connection fails
            
        # reset
   
        remote_dir = "/home/ubuntu/V22_instance/CAC-DAS_bilevel/Tune_T=500_R=100/"
        delete_in_folder(hostname, port, username, key_filepath, remote_dir)
        