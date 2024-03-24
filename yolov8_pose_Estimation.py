#  POSE ESTIMATION..
import numpy as np
import torch
from ultralytics import YOLO

# Load a pretrained Pose model
model = YOLO('yolov8m-pose.pt')
    
    
import cv2

# Function to load and process a frame
def process_frame(frame):
  results = model.predict(frame)  # Assuming model predicts keypoints

  for r in results:
    xy = r.keypoints.xy  # Extract keypoints
    return xy


# Function to analyze action based on keypoint changes
def analyze_action(prev_keypoints, current_keypoints):
  # Keypoint indices
  nose_idx = 0
  left_eye_idx = 1
  right_eye_idx = 2
  left_ear_idx = 3
  right_ear_idx = 4
  left_shoulder_idx = 5
  right_shoulder_idx = 6
  left_elbow_idx = 7
  right_elbow_idx = 8
  left_wrist_idx = 9
  right_wrist_idx = 10
  left_hip_idx = 11
  right_hip_idx = 12
  left_knee_idx = 13
  right_knee_idx = 14
  left_ankle_idx = 15
  right_ankle_idx = 16


  # Thresholds (adjust as needed based on your data)
  punch_threshold = 30  # Adjust for significant wrist movement in x-axis
  jump_threshold = 25  # Adjust for vertical movement in knees and ankles
  kick_threshold = 25  # Adjust for significant movement in one leg's ankle while the other remains still
  walk_threshold = 30
  run_threshold = 80  # Adjust for alternating significant movement in knees and ankles
  stillness_threshold = 10  # Adjust for minimal movement
  
  
  # Punching detection
  shoulder_change_jump = abs(current_keypoints[0][left_shoulder_idx][1] - prev_keypoints[0][left_shoulder_idx][1]) + \
                        abs(current_keypoints[0][right_shoulder_idx][1] - prev_keypoints[0][right_shoulder_idx][1])
  left_wrist_change_x = abs(current_keypoints[0][left_wrist_idx][0] - prev_keypoints[0][left_wrist_idx][0])
  right_wrist_change_x = abs(current_keypoints[0][right_wrist_idx][0] - prev_keypoints[0][right_wrist_idx][0])

#   is_likely_punching = (left_wrist_change_x > punch_threshold or right_wrist_change_x > punch_threshold) and (shoulder_change_jump <= stillness_threshold)
#   left_elbow_change_x = abs(current_keypoints[0][left_elbow_idx][0] - prev_keypoints[0][left_elbow_idx][0])
#   right_elbow_change_x = abs(current_keypoints[0][right_elbow_idx][0] - prev_keypoints[0][right_elbow_idx][0])

  is_likely_punching =  (left_wrist_change_x > punch_threshold or right_wrist_change_x > punch_threshold) 


  
  # Jumping detection...
  
  shoulder_change_jump_y = abs(current_keypoints[0][left_shoulder_idx][1] - prev_keypoints[0][left_shoulder_idx][1]) + \
                        abs(current_keypoints[0][right_shoulder_idx][1] - prev_keypoints[0][right_shoulder_idx][1])
  shoulder_change_jump_x = abs(current_keypoints[0][left_shoulder_idx][0] - prev_keypoints[0][left_shoulder_idx][0]) + \
                        abs(current_keypoints[0][right_shoulder_idx][0] - prev_keypoints[0][right_shoulder_idx][0])
  knee_change_jump_y = abs(current_keypoints[0][left_knee_idx][1] - prev_keypoints[0][left_knee_idx][1]) + \
                        abs(current_keypoints[0][right_knee_idx][1] - prev_keypoints[0][right_knee_idx][1])
  ankle_change_jump_y = abs(current_keypoints[0][left_ankle_idx][1] - prev_keypoints[0][left_ankle_idx][1]) + \
                        abs(current_keypoints[0][right_ankle_idx][1] - prev_keypoints[0][right_ankle_idx][1])


#   keypoint_changes = abs(current_keypoints - prev_keypoints)[:, 1:2]  # Extract y changes 
#   avg_velocity_y_jumping = torch.mean(keypoint_changes[:, 0])
   
  is_likely_jumping = knee_change_jump_y > jump_threshold and ankle_change_jump_y > jump_threshold and shoulder_change_jump_y > jump_threshold 



  # Kicking detection
  left_ankle_change_kick_y = abs(current_keypoints[0][left_ankle_idx][1] - prev_keypoints[0][left_ankle_idx][1])
  right_ankle_change_kick_y = abs(current_keypoints[0][right_ankle_idx][1] - prev_keypoints[0][right_ankle_idx][1])
  
  left_ankle_change_kick_x = abs(current_keypoints[0][left_ankle_idx][0] - prev_keypoints[0][left_ankle_idx][0])
  right_ankle_change_kick_x = abs(current_keypoints[0][right_ankle_idx][0] - prev_keypoints[0][right_ankle_idx][0])
  
  is_likely_kicking = (left_ankle_change_kick_y > kick_threshold and left_ankle_change_kick_x > kick_threshold and right_ankle_change_kick_y < stillness_threshold) or (right_ankle_change_kick_y > kick_threshold and right_ankle_change_kick_x > kick_threshold and left_ankle_change_kick_y < stillness_threshold)

  
  
  
#   Walking and Running detection...
  
  knee_change_run = abs(current_keypoints[0][left_knee_idx][1] - prev_keypoints[0][left_knee_idx][1]) + \
                    abs(current_keypoints[0][right_knee_idx][1] - prev_keypoints[0][right_knee_idx][1])
  ankle_change_run = abs(current_keypoints[0][left_ankle_idx][1] - prev_keypoints[0][left_ankle_idx][1]) + \
                     abs(current_keypoints[0][right_ankle_idx][1] - prev_keypoints[0][right_ankle_idx][1])
    
  is_likely_walking = knee_change_run > walk_threshold or  ankle_change_run > walk_threshold
    
    
#   # Calculate absolute difference for each keypoint (assuming x and y at indices 0 and 1)
#   keypoint_changes = abs(current_keypoints - prev_keypoints)[:, :1]  # Extract x changes
# #   print(type(keypoint_changes))
# #   print(type(keypoint_changes[:, 0]))
  
#   # Average absolute change in x and y
#   avg_velocity_x = torch.mean(keypoint_changes[:, 0])
# #   avg_velocity_y = torch.mean(keypoint_changes[:, 1])
    
#   is_likely_walking_running = (avg_velocity_x > walk_threshold and avg_velocity_x < run_threshold)   
  

  # Running detection
  knee_change_run = abs(current_keypoints[0][left_knee_idx][1] - prev_keypoints[0][left_knee_idx][1]) + \
                    abs(current_keypoints[0][right_knee_idx][1] - prev_keypoints[0][right_knee_idx][1])
  ankle_change_run = abs(current_keypoints[0][left_ankle_idx][1] - prev_keypoints[0][left_ankle_idx][1]) + \
                     abs(current_keypoints[0][right_ankle_idx][1] - prev_keypoints[0][right_ankle_idx][1])
  is_likely_running = knee_change_run > run_threshold or ankle_change_run > run_threshold
  
  

  return is_likely_punching, is_likely_jumping, is_likely_kicking, is_likely_walking, is_likely_running



# Combine with other cues for stronger prediction (optional)
# - Consider hip movement
# - Consider torso movement





# Drawing Bounding Boxes...

def draw_bounding_box(frame, keypoints):
  # Assuming keypoints[0] contains all keypoints for a single person
  # Identify keypoints for top left and bottom right corners (adjust based on your model)
  top_left_x = int(keypoints[0][:, 0].min().item())  # Minimum x-coordinate across all keypoints
  top_left_y = int(keypoints[0][:, 1].min().item())-20  # Minimum y-coordinate across all keypoints
  bottom_right_x = int(keypoints[0][:, 0].max().item())  # Maximum x-coordinate across all keypoints
  bottom_right_y = int(keypoints[0][:, 1].max().item())+20 # Maximum y-coordinate across all keypoints

  # Define bounding box thickness and color
  thickness = 2
  color = (0, 255, 0)  # Green for bounding box

  # Draw the rectangle
  cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

  return frame




# Display action label...

def display_action_label(frame, action, confidence):
  # Define text font, scale, color, and thickness
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.7
  text_color = (0, 0, 255)  # Red for text
  text_thickness = 2

  # Get frame dimensions for text placement
  (text_width, text_height) = cv2.getTextSize(action, font, font_scale, text_thickness)[0]

  # Adjust text position based on your preference (e.g., top left corner)
  text_offset_x = 10
  text_offset_y = 20

  # Display action and confidence score
  cv2.putText(frame, f"{action}: {confidence:.2f}", (text_offset_x, text_offset_y), font, font_scale, text_color, text_thickness)

  return frame






# List to store keypoints from multiple frames
keypoints_list = []

# Initialize running confidence
punching_confidence = 0
jumping_confidence = 0
kicking_confidence = 0
running_confidence = 0
walking_confidence = 0


# Load video capture object
cap = cv2.VideoCapture("C:\\Users\\ag701\\Desktop\\Machine Learning\\Annotated_images\\jumping1 (3).mp4")

# # Initialize webcam capture
# cap = cv2.VideoCapture(0)  # Change to index for other cameras


# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame and get keypoints
    keypoints = process_frame(frame)
    keypoints_list.append(keypoints)
    
    # print(keypoints_list)

    # Analyze consecutive frames
    if len(keypoints_list) > 1:
        prev_keypoints = keypoints_list[-2]
        current_keypoints = keypoints_list[-1]
        
        # print(current_keypoints[0][0][1])
        is_likely_punching, is_likely_jumping, is_likely_kicking, is_likely_walking, is_likely_running = analyze_action(prev_keypoints, current_keypoints)
        
        
        # Update running confidence based on analysis
        if is_likely_punching:
            punching_confidence += 0.2  # Adjust confidence update as needed
        else:
            punching_confidence -= 0.1

        punching_confidence = max(0, min(punching_confidence, 1))  # Keep confidence within 0-1
        
        # Display prediction (consider adding bounding boxes or visualizations)
        
        # print("Punching confidence:", punching_confidence)
        # cv2.imshow('Punching Detection', frame)
        
        
        
        if is_likely_jumping:
            jumping_confidence += 0.2  # Adjust confidence update as needed
        else:
            jumping_confidence -= 0.1

        jumping_confidence = max(0, min(jumping_confidence, 1))  # Keep confidence within 0-1
        
        # Display prediction (consider adding bounding boxes or visualizations)
        
        # print("Jumping confidence:", jumping_confidence)
        # cv2.imshow('Jumping Detection', frame)
        
        
        
        
        if is_likely_kicking:
            kicking_confidence += 0.2  # Adjust confidence update as needed
        else:
            kicking_confidence -= 0.1

        kicking_confidence = max(0, min(kicking_confidence, 1))  # Keep confidence within 0-1
        
        # Display prediction (consider adding bounding boxes or visualizations)

        # print("kicking confidence:", kicking_confidence)
        # cv2.imshow('Kicking Detection', frame)
        
        
        
        
        if is_likely_walking:
            walking_confidence += 0.2  # Adjust confidence update as needed
        else:
            running_confidence -= 0.1
            
        walking_confidence = max(0, min(walking_confidence, 1))  # Keep confidence within 0-1
        
        
        if is_likely_running:
            running_confidence += 0.2
        else:
            running_confidence -= 0.1
            
        running_confidence = max(0, min(running_confidence, 1))
        
        # Display prediction (consider adding bounding boxes or visualizations)
        
        # print("Walking confidence:", walking_confidence)
        # cv2.imshow('Running Detection', frame)
        
        
        
        # if is_likely_running:
        #     running_confidence += 0.2  # Adjust confidence update as needed
        # else:
        #     running_confidence -= 0.1

        # running_confidence = max(0, min(running_confidence, 1))  # Keep confidence within 0-1
        
        # Display prediction (consider adding bounding boxes or visualizations)
        
        # print("Running confidence:", running_confidence)
        # cv2.imshow('Running Detection', frame)
            
            
        maximum_confidence = max(punching_confidence, jumping_confidence, kicking_confidence, walking_confidence, running_confidence)
        
        if maximum_confidence == punching_confidence:
            print('Punch Detected: ', punching_confidence)
            # cv2.imshow('Punch Detection', frame)
            maximum_conf_action = 'punching'
            
            
        elif maximum_confidence == jumping_confidence:
            print('Jump Detected: ', jumping_confidence)
            # cv2.imshow('Jump Detection', frame)
            maximum_conf_action = 'jumping'
            
        elif maximum_confidence == kicking_confidence:
            print('Kick Detected: ', kicking_confidence)
            # cv2.imshow('Kick Detection', frame)
            maximum_conf_action = 'kicking'
            
            
        elif maximum_confidence == walking_confidence:
            print('Walking Detected: ', walking_confidence)
            # cv2.imshow('Walk Detection', frame)
            maximum_conf_action = 'walking'
            
            
        else:
            print('Running Detected: ', running_confidence)
            # cv2.imshow('Running Detection', frame)
            maximum_conf_action = 'running'
            
            
            
        # Draw bounding box and display action label
        frame = draw_bounding_box(frame, current_keypoints)
        frame = display_action_label(frame, maximum_conf_action, maximum_confidence)
        
        
        # Display the frame with bounding box and action label
        cv2.imshow('Action Detection', frame)
            
            
            
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

