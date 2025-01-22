# %% [markdown]
# Import and Install Dependencies

# %%
!pip install tensorflow==2.17.1 opencv-python mediapipe scikit-learn matplotlib


# %%
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# %% [markdown]
#  Keypoints using MP Holistic

# %%
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# %%
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# %%
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

# %%
def draw_styled_landmarks(image, results):
    # Draw face connections using FACEMESH_TESSELATION
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


# %%
import cv2
import mediapipe as mp

# Initialize mediapipe drawing and holistic modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

# Function for mediapipe detection
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_rgb, results

# Function to draw landmarks with specific styling
def draw_styled_landmarks(image, results):
    # Draw face connections using FACEMESH_TESSELATION
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


# %%
if results.left_hand_landmarks:
    # If landmarks are detected, access and process them
    num_landmarks = len(results.left_hand_landmarks.landmark)
else:
    # If no landmarks are detected, handle it gracefully
    num_landmarks = 0  # or any default value you prefer


# %%
results

# %%
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Add this for face mesh connections

# Function to draw landmarks
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)  # Updated for face connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


# %%
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# %% [markdown]
# 3. Extract Keypoint Values

# %%
if results.left_hand_landmarks:
    left_hand_landmarks = results.left_hand_landmarks.landmark
    print(len(left_hand_landmarks))
else:
    print("Left hand landmarks not detected.")


# %%
pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)

# %%
pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

# %%
face = (
    np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    if results.face_landmarks
    else np.zeros(1404)
)


# %%
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# %%
result_test = extract_keypoints(results)

# %%
result_test

# %%
np.save('0', result_test)

# %%
np.load('0.npy')

# %% [markdown]
# Setup Folder for colllection

# %%
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello','please','thanks','no','sorry'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

# %%
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# %% [markdown]
# 5. Collect Keypoint Values for Training and Testing

# %%
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
                    
    cap.release()
    cv2.destroyAllWindows()

# %% [markdown]
# 6. Preprocess Data and Create Labels and Features

# %%
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# %%
label_map = {label:num for num, label in enumerate(actions)}

# %%
label_map

# %%
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# %%
np.array(sequences).shape

# %%
np.array(labels).shape

# %%
X = np.array(sequences)

# %%
X.shape

# %%
y = to_categorical(labels).astype(int)

# %%
y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# %%
y_train.shape

# %% [markdown]
# 7. Build and Train LSTM Neural Network

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# %%
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# %%
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# %%
res = [.7, 0.2, 0.1]

# %%

actions[np.argmax(res)]

# %%
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# %%

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# %%
model.summary()

# %%
model.evaluate(X_test,y_test)

# %%
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import numpy as np

# Assuming model.predict() and data processing are done

# Get predictions and true values
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1)  # True labels
yhat_classes = np.argmax(yhat, axis=1)  # Predicted labels

# Calculate metrics
precision = precision_score(ytrue, yhat_classes, average='weighted')
recall = recall_score(ytrue, yhat_classes, average='weighted')
f1 = f1_score(ytrue, yhat_classes, average='weighted')
accuracy = accuracy_score(ytrue, yhat_classes)

# Print metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Check unique classes
print("Unique classes in ytrue:", np.unique(ytrue))
print("Unique classes in yhat_classes:", np.unique(yhat_classes))

# Adjust target_names to match the number of classes
actions = ['Action 1', 'Action 2', 'Action 3', 'Action 4']  # Update this list

# Print classification report
print("\nClassification Report:")
print(classification_report(ytrue, yhat_classes, target_names=actions))


# %% [markdown]
# 8. Make Predictions

# %%
res = model.predict(X_test)

# %%
actions[np.argmax(res[0])]

# %%
actions[np.argmax(y_test[0])]

# %% [markdown]
# 

# %%
model.save('action.h5')

# %%
model.load_weights('action.h5')

# %%


# %% [markdown]
# 10. Evaluation using Confusion Matrix and Accuracy

# %%
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
yhat = model.predict(X_test)

# %%
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# %%

multilabel_confusion_matrix(ytrue, yhat)

# %%
cm = confusion_matrix(ytrue, yhat_classes)
print("Confusion Matrix:\n", cm)

# %%
print(classification_report(ytrue, yhat_classes, target_names=actions))

# %%
accuracy = accuracy_score(ytrue, yhat_classes)
print("Accuracy:", accuracy)

# %%
accuracy_score(ytrue, yhat)

# %% [markdown]
# 11. Test in Real Time

# %%
from scipy import stats

# %%
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# %%
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Assuming these variables are defined somewhere in your code
actions = ['hello','please','thanks','no','sorry']  # example action labels
colors = [(245, 117, 16), (0, 255, 0), (0, 0, 255), (255, 0, 0),(82,74,45)]  # example colors

# Example prob_viz function
def prob_viz(res, actions, image, colors):
    output_frame = image.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Simulated result (probabilities from model prediction)
res = [0.9, 0.1, 0.05, 0.2,0.9]  # Example prediction probabilities for each action

# Simulated image (use actual frame from webcam)
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image for testing

# Visualize the result
plt.figure(figsize=(18, 18))
plt.imshow(prob_viz(res, actions, image, colors))
plt.axis('off')  # Turn off axis if you don't want to see axis numbers
plt.show()


# %%
from scipy.special import softmax

sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            res = softmax(res)  # Apply softmax for confidence scores
            predictions.append(np.argmax(res))
            
            # Display top-3 predictions with confidence
            top_3 = np.argsort(res)[-3:][::-1]
            print("Top Predictions:")
            for idx in top_3:
                print(f"{actions[idx]}: {res[idx]:.2f}")

            # Update sentence
            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualization
            image = prob_viz(res, actions, image, colors)
        
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


