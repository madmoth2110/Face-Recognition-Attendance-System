\# Face Recognition Attendance System



A Python and OpenCV based Face Recognition Attendance System for employee registration, face recognition, emotion estimation, and automatic attendance management.



\## Features



\- Employee registration using webcam

\- Multiple face samples for better recognition

\- Face recognition using LBPH

\- Real-time face detection

\- Emotion detection using DeepFace

\- Automatic attendance marking

\- Date and time recording

\- Attendance viewing through GUI

\- Simple Tkinter-based interface



\## Technologies Used



\- Python

\- OpenCV

\- DeepFace

\- TensorFlow

\- Tkinter

\- CSV

\- LBPH Face Recognizer

\- Haar Cascade Classifier



\## Project Structure



| File | Description |

|------|-------------|

| `register.py` | Registers employees and captures face samples |

| `train.py` | Trains the LBPH face recognition model |

| `recognize.py` | Recognizes faces, estimates emotions, and marks attendance |

| `emotion\_test.py` | Tests emotion detection using DeepFace |

| `gui.py` | Provides the graphical user interface |

| `main.py` | Main project module |

| `face.py` | Face-related functionality |

| `motion.py` | Motion detection functionality |

| `security\_camera.py` | Security camera functionality |

| `nisha.py` | Additional project functionality |



\## How It Works



1\. Register an employee using the webcam.

2\. Capture multiple face samples.

3\. Train the LBPH face recognition model.

4\. Start the face recognition system.

5\. The system detects and recognizes registered faces.

6\. DeepFace estimates the person's facial expression.

7\. Attendance is automatically recorded with date and time.

8\. Attendance records can be viewed through the GUI.



\## Installation



Install the required packages:



```bash

pip install opencv-contrib-python

pip install deepface

pip install tf-keras

