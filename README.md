# Face Recognition Attendance System

A Python and OpenCV based Face Recognition Attendance System for employee registration, face recognition, and automatic attendance management.

## Features

* Employee registration using webcam
* Multiple face samples for better recognition
* Face recognition using LBPH
* Real-time face detection
* Automatic attendance marking
* Date and time recording
* Attendance viewing through GUI
* Simple Tkinter-based interface

## Technologies Used

* Python
* OpenCV
* Tkinter
* CSV
* LBPH Face Recognizer
* Haar Cascade Classifier

## Project Structure

| File                 | Description                                   |
| -------------------- | --------------------------------------------- |
| `register.py`        | Registers employees and captures face samples |
| `train.py`           | Trains the LBPH face recognition model        |
| `recognize.py`       | Recognizes faces and marks attendance         |
| `gui.py`             | Provides the graphical user interface         |
| `main.py`            | Main project module                           |
| `face.py`            | Face-related functionality                    |
| `motion.py`          | Motion detection functionality                |
| `security_camera.py` | Security camera functionality                 |
| `nisha.py`           | Additional project functionality              |

## How It Works

1. Register an employee using the webcam.
2. Capture multiple face samples.
3. Train the face recognition model.
4. Start the camera.
5. The system detects and recognizes registered faces.
6. Attendance is automatically recorded with date and time.
7. Attendance records can be viewed through the GUI.

## Installation

Install the required OpenCV package:

```bash
pip install opencv-contrib-python
```

## How to Run

Start the graphical interface:

```bash
python gui.py
```

From the GUI, you can:

* Register an employee
* Train the model
* Start face recognition
* View attendance records
* Exit the application

## Project Screenshot

![Face Recognition System](./screenshot.png)

## Privacy Note

This project is created for learning and demonstration purposes.

Real employee face images and attendance records contain personal data and should not be uploaded to a public repository.

## Future Improvements

* Improve recognition accuracy
* Add a database for employee records
* Add login/authentication
* Generate attendance reports
* Add an admin dashboard
* Improve the GUI design

## License

This project is intended for educational and demonstration purposes.
