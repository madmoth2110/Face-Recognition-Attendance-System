import cv2
import os
import numpy as np
import json

path = "employees"

faces = []
ids = []
names = {}

employee_names = set()

# Find all employee names
for file in os.listdir(path):

    if not file.lower().endswith(".jpg"):
        continue

    base_name = os.path.splitext(file)[0]

    # Example: Nisha_0 -> Nisha
    if "_" in base_name and base_name.rsplit("_", 1)[1].isdigit():
        employee_name = base_name.rsplit("_", 1)[0]
    else:
        employee_name = base_name

    employee_names.add(employee_name)


# Assign one ID to each employee
for current_id, employee_name in enumerate(
    sorted(employee_names, key=str.lower)
):

    names[current_id] = employee_name

    for file in os.listdir(path):

        if not file.lower().endswith(".jpg"):
            continue

        base_name = os.path.splitext(file)[0]

        # Get employee name from filename
        if "_" in base_name and base_name.rsplit("_", 1)[1].isdigit():
            file_employee = base_name.rsplit("_", 1)[0]
        else:
            file_employee = base_name

        if file_employee.lower() != employee_name.lower():
            continue

        # Read the saved face image directly
        image = cv2.imread(
            os.path.join(path, file),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        faces.append(image)
        ids.append(current_id)


# Check if images were found
if len(faces) == 0:
    print("No face images found.")
    exit()


# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(ids)
)

# Save trained model
recognizer.save("trainer.yml")

# Save employee names
with open("names.json", "w") as file:
    json.dump(names, file)

print("Training completed")
print(names)
print("Total training images:", len(faces))