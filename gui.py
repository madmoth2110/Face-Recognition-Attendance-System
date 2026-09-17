import tkinter as tk
import subprocess
import csv
from tkinter import ttk

window = tk.Tk()
window.title("Face Recognition System")
window.geometry("600x600")

title = tk.Label(window, text="FACE RECOGNITION SYSTEM", font=("Arial", 24, "bold"))
title.pack(pady=30)

subtitle = tk.Label(window, text="Employee Attendance Management", font=("Arial", 12))
subtitle.pack(pady=5)

def start_camera():
    subprocess.run(["python", "recognize.py"])

start_button = tk.Button(window, text="START CAMERA", width=30, height=2, command=start_camera)
start_button.pack(pady=10)

def register_employee():
    subprocess.run(["python", "register.py"])

register_button = tk.Button(window, text="REGISTER EMPLOYEE", width=30, height=2, command=register_employee)
register_button.pack(pady=10)

def train_model():
    subprocess.run(["python", "train.py"])

train_button = tk.Button(window, text="TRAIN MODEL", width=30, height=2, command=train_model)
train_button.pack(pady=10)

def view_attendance():
    attendance_window = tk.Toplevel(window)
    attendance_window.title("Attendance Record")
    attendance_window.geometry("700x450")

    title = tk.Label(attendance_window, text="ATTENDANCE RECORD", font=("Arial", 20, "bold"))
    title.pack(pady=20)

    table = ttk.Treeview(attendance_window, columns=("Name", "Date", "Time"), show="headings")

    table.heading("Name", text="Name")
    table.heading("Date", text="Date")
    table.heading("Time", text="Time")

    table.column("Name", width=220)
    table.column("Date", width=220)
    table.column("Time", width=220)

    table.pack(pady=15)

    try:
        with open("attendance.csv", "r") as file:
            reader = csv.DictReader(file)

            for row in reader:
                table.insert("", "end", values=(row["Name"], row["Date"], row["Time"]))

    except FileNotFoundError:
        table.insert("", "end", values=("No attendance", "-", "-"))

attendance_button = tk.Button(window, text="VIEW ATTENDANCE", width=30, height=2, command=view_attendance)
attendance_button.pack(pady=10)

def exit_program():
    window.destroy()

exit_button = tk.Button(window, text="EXIT", width=30, height=2, command=exit_program)
exit_button.pack(pady=10)

window.mainloop()