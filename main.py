import argparse
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading

def run_unsupervised():
    output_text.insert(tk.END, "\n[INFO] Running unsupervised analysis...\n")
    root.update_idletasks()
    from unsupervised_main import main as unsupervised_main
    unsupervised_main()
    output_text.insert(tk.END, "[SUCCESS] Unsupervised analysis complete.\n")


def run_supervised():
    output_text.insert(tk.END, "\n[INFO] Running supervised prediction...\n")
    root.update_idletasks()
    from supervised_model import main as supervised_main
    supervised_main()
    output_text.insert(tk.END, "[SUCCESS] Supervised prediction complete.\n")


def threaded_run(func):
    thread = threading.Thread(target=func)
    thread.start()

def main():
    global root, output_text
    root = tk.Tk()
    root.title("Web Pattern Profiling")
    root.geometry("500x400")

    label = tk.Label(root, text="Select a mode:", font=("Arial", 14))
    label.pack(pady=10)

    btn_unsupervised = tk.Button(root, text="Run Unsupervised Analysis", command=lambda: threaded_run(run_unsupervised), width=40)
    btn_unsupervised.pack(pady=5)

    btn_supervised = tk.Button(root, text="Run Supervised Prediction", command=lambda: threaded_run(run_supervised), width=40)
    btn_supervised.pack(pady=5)

    btn_exit = tk.Button(root, text="Exit", command=root.quit, width=40)
    btn_exit.pack(pady=10)

    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, width=60)
    output_text.pack(pady=10)
    output_text.insert(tk.END, "[READY] Select a task to begin...\n")

    root.mainloop()

if __name__ == "__main__":
    main()
