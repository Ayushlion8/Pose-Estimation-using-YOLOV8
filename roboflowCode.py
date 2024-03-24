from roboflow import Roboflow
import cv2
from yolov8_pose_Estimation import process_frame,analyze_action,draw_bounding_box,display_action_label

rf = Roboflow(api_key="5Gcc90YOduXlMZo2GX1X")
project = rf.workspace().project("har-dl6ob")
#Specify your ROboflow project version
model = project.version(1).model

#Specify your video source, usually if you have only a webcam your source is 0 so cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)
# Load video capture object
cap = cv2.VideoCapture("production_id_5025965 (1080p).mp4")
# List to store keypoints from multiple frames
keypoints_list = []

# Initialize running confidence
punching_confidence = 0
jumping_confidence = 0
kicking_confidence = 0
running_confidence = 0
walking_confidence = 0


while True:
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 640))
    result = model.predict(frame, confidence=50, overlap=30).json()        
   
    for prediction in result['predictions']:
        x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
        label = prediction['class']
        if label == "Lying":
            label = "standing"
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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
        
            
    # Show screen with your predictions
    cv2.imshow("Roboflow realtime detection", frame)
    
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free webcam
cap.release()
cv2.destroyAllWindows()