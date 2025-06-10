
import tkinter as tk
from tkinter import filedialog
import subprocess
import os

def decrypt_file():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    with open("kc.txt", "r") as f:
        word = f.read().strip()
    output_path = file_path.replace(".enc", "")
    subprocess.run([
        "openssl", "enc", "-d", "-aes-256-cbc",
        "-in", file_path,
        "-out", output_path,
        "-pass", f"pass:{word}"
    ])
    print(f"Decrypted to {output_path}")

root = tk.Tk()
tk.Button(root, text="Decrypt File", command=decrypt_file).pack(padx=20, pady=20)
root.mainloop()