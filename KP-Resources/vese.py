
import tkinter as tk
from tkinter import filedialog
import subprocess
import os



def grabPass(): # not using this anymore.
    with open("kc.txt", "r") as file:
        word = file.read().strip()
    return word


def encryptFile(): # Function to Encrypt the given passed file.
    """
    This whole script's point is not really top of the line security, its a quick and dirty encryption of files to transfer
    between computers. 
    
    """
    
    file_path = filedialog.askopenfilename(title="Select a file to encrypt") # choose a file to make secret
    if not file_path: # you failed
        return
    
    with open("kc.txt", "r") as f: # the password is hardset, to avoid typing it in a lot.
        word = f.read().strip()
    output_path = file_path + ".enc"
    
    subprocess.run([ # call a subprocess to run terminal commands. 
        "openssl", "enc", "-aes-256-cbc", "-salt",
        "-in", file_path,
        "-out", output_path,
        "-pass", f"pass:{word}"
    ])
    print(f"Compressed to {output_path}")
    
if __name__ == "__main__":
    root = tk.Tk()
    tk.Button(root, text="Encrypt File", command=encryptFile).pack(padx = 20, pady = 20)
    root.mainloop()



