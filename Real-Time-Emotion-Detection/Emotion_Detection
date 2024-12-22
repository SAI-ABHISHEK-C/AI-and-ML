import os
import warnings
from deepface import DeepFace
import cv2

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the webcam.")
        break

    try:
        # Analyze the frame for emotions, age, and gender
        results = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)

        # Debugging: Print the output structure
        # Uncomment the next line to see the full output from DeepFace
        # print(results)

        # Check if the result is a list and handle accordingly
        if isinstance(results, list):
            results = results[0]  # Access the first element in the list

        # Extract face bounding box, age, and emotions
        region = results.get("region", {})  # Get face bounding box
        dominant_emotion = results.get("dominant_emotion", "Unknown")  # Extract dominant emotion
        age = results.get("age", "Unknown")  # Extract age

        # Handle gender confidence
        gender_confidence = results.get("gender", {})  # Gender confidence dictionary
        if isinstance(gender_confidence, dict):
            man_conf = gender_confidence.get("Man", 0)
            woman_conf = gender_confidence.get("Woman", 0)

            # Determine gender if confidence is greater than 60%
            if man_conf > 60:
                gender = "Man"
            elif woman_conf > 60:
                gender = "Woman"
            else:
                gender = "Unknown"
        else:
            gender = "Unknown"

        # Draw a bounding box around the face
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add a label above the bounding box
        label = f"{gender}, {dominant_emotion}, Age: {int(age)}"
        cv2.putText(
            frame,
            label,
            (x, y - 10),  # Position the label above the bounding box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (36, 255, 12),
            2,
        )

    except Exception as e:
        print(f"Error during analysis: {e}")

    # Show the frame with bounding box and details
    cv2.imshow('Age, Gender, and Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
