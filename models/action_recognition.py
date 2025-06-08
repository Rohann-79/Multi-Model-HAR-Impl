import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorchvideo.models.resnet
import numpy as np
import cv2
from typing import Dict, List
import json
import os
from collections import deque, Counter
import time
import torch.nn.functional as F

class ActionRecognitionModel:
    def __init__(self, model_name: str = "i3d_r50", num_classes: int = 400):
        """Initialize the action recognition model"""
        # Force CPU usage for M1 Mac compatibility
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe Pose with higher detection confidence
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize MediaPipe hands for better drinking detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize temporal smoothing
        self.prediction_history = deque(maxlen=5)  # Keep last 5 predictions
        self.confidence_threshold = 0.01  # Very low threshold for demo
        
        # Store positions for movement and detection
        self.shoulder_positions = deque(maxlen=10)  # Store last 10 x positions
        self.shoulder_heights = deque(maxlen=10)    # Store last 10 y positions
        self.vertical_velocities = deque(maxlen=5)  # Store velocity history for fall detection
        self.last_fall_detection = 0  # Time of last fall detection
        self.last_drink_detection = 0  # Time of last drink detection
        
        # Focus on activities
        self.relevant_activities = {
            'sitting',
            'standing',
            'walking',
            'running',
            'eating',
            'drinking',
            'looking at phone',
            'fighting',
            'falling'
        }
        
        # Load model and transforms
        self.model = self._load_model(model_name, num_classes)
        self.transform = self._get_transform()
        self.class_names = self._load_class_names()
        print(f"Loaded {len(self.class_names)} activity classes")
        print("Monitoring for activities:", list(self.relevant_activities))
        
        # Map simple actions to model class indices (after class_names loaded)
        self.simple_action_map = {
            'sitting': ['sitting'],
            'standing': ['standing'],
            'walking': ['walking'],
            'running': ['running'],
            'drinking': ['drinking'],
            'eating': ['eating', 'eating food', 'feeding'],
            'looking at phone': ['texting message', 'mobile phone', 'looking at phone', 'using phone'],
            'fighting': ['fighting', 'wrestling', 'boxing', 'sparring', 'karate', 'judo', 'punching'],
            'falling': ['falling', 'tripping', 'slipping', 'fainting']
        }
        self.class_indices_map = {}
        for label, keywords in self.simple_action_map.items():
            self.class_indices_map[label] = [
                i for i, cn in enumerate(self.class_names)
                if any(kw in cn.lower() for kw in keywords)
            ]
        print("Simple action class indices map:", {k: len(v) for k, v in self.class_indices_map.items()})
        
    def _load_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Load the PyTorchVideo model"""
        try:
            # Use I3D model which is better for close-up views
            model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
            print("Successfully loaded I3D model")
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading I3D model: {e}")
            print("Falling back to Slow R50 model...")
            try:
                model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading fallback model: {e}")
                return None
    
    def _get_transform(self) -> transforms.Compose:
        """Get video preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_class_names(self) -> List[str]:
        """Load action class names"""
        try:
            kinetics_labels = torch.hub.load('facebookresearch/pytorchvideo', 'kinetics_400_labels')
            # Save labels to file for reference
            with open("models/kinetics400_classes.json", "w") as f:
                json.dump(kinetics_labels, f, indent=4)
            return kinetics_labels
        except:
            if os.path.exists("models/kinetics400_classes.json"):
                with open("models/kinetics400_classes.json", "r") as f:
                    return json.load(f)
            return []
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame"""
        try:
            frame_tensor = self.transform(frame)
            return frame_tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None
    
    def _analyze_pose(self, frame: np.ndarray) -> str:
        """Analyze pose to determine activities"""
        # Get original frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Process frame with MediaPipe for pose detection
        results_pose = self.pose.process(frame)
        
        # Check for drinking water using both pose and blob detection
        is_drinking = self._detect_drinking(frame, results_pose)
        if is_drinking:
            return 'drinking'
        
        # If no pose landmarks detected, return None
        if not results_pose.pose_landmarks:
            return None
            
        landmarks = results_pose.pose_landmarks.landmark
        
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE] 
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate positions
        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
        hip_height = (left_hip.y + right_hip.y) / 2
        knee_height = (left_knee.y + right_knee.y) / 2
        ankle_height = (left_ankle.y + right_ankle.y) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # Store positions for movement detection
        self.shoulder_positions.append(shoulder_x)
        self.shoulder_heights.append(shoulder_height)
        
        # Check for fall directly - takes precedence over other activities
        is_falling = self._detect_fall_position(landmarks, frame_height, frame_width)
        if is_falling:
            return 'falling'
        
        # Determine basic posture (sitting or standing)
        posture = self._determine_posture(shoulder_height, hip_height, knee_height)
        
        # Check for looking at phone when sitting
        if posture == "sitting":
            is_looking_at_phone = self._detect_looking_at_phone(frame, landmarks)
            if is_looking_at_phone:
                return 'looking at phone'
        
        print(f"Debug - Shoulder height: {shoulder_height:.2f}, Hip height: {hip_height:.2f}")
        
        # Enhanced fall detection based on movement
        if len(self.shoulder_heights) >= 5:
            recent_heights = list(self.shoulder_heights)[-5:]
            vertical_velocity = (recent_heights[-1] - recent_heights[0]) / 5
            self.vertical_velocities.append(vertical_velocity)
            
            # Get current posture - important for fall vs. sitting detection
            current_posture = self._determine_posture(shoulder_height, hip_height, knee_height)
            
            # Calculate statistics for better fall detection
            avg_velocity = sum(self.vertical_velocities) / len(self.vertical_velocities)
            max_velocity = max(self.vertical_velocities) if self.vertical_velocities else 0
            final_position = recent_heights[-1]
            
            print(f"Debug - Current posture: {current_posture}")
            print(f"Debug - Vertical velocity: {vertical_velocity:.3f}, Max velocity: {max_velocity:.3f}")
            print(f"Debug - Final shoulder position: {final_position:.2f}")
            
            current_time = time.time()
            # Improved fall detection criteria:
            # 1. Rapid downward movement (high velocity)
            # 2. Final position is very low in the frame
            # 3. NOT in sitting posture (to avoid triggering when just sitting down)
            # 4. At least 5 seconds since last detection
            if (max_velocity > 0.08 and       # Rapid downward movement
                final_position > 0.7 and      # Ended very low in frame (lower threshold than sitting)
                current_posture != "sitting" and # Not in sitting posture
                current_time - self.last_fall_detection > 5):
                
                self.last_fall_detection = current_time
                print("FALL DETECTED!")
                return 'falling'
        
        # For movement detection (walking/running)
        if len(self.shoulder_positions) >= 5:
            x_positions = list(self.shoulder_positions)
            movement = max(x_positions) - min(x_positions)
            
            print(f"Debug - Movement amount: {movement:.3f}")
            
            # Only check for walking/running if in standing posture
            if posture == "standing":
                if movement > 0.05:
                    return 'walking'
        
        # Return the basic posture (sitting or standing)
        return posture
        
    def _determine_posture(self, shoulder_height, hip_height, knee_height):
        """
        Determine if the person is sitting or standing based on joint relationships
        
        In a standing pose:
        - Shoulders are significantly higher than hips
        - Hips are significantly higher than knees
        - Shoulders are in the upper part of the frame
        
        In a sitting pose:
        - Shoulders and hips are closer together vertically
        - Knees might be higher relative to hips
        - Shoulder position is in middle part of frame
        """
        # Calculate joint relationships
        shoulder_hip_distance = hip_height - shoulder_height
        hip_knee_distance = knee_height - hip_height
        
        # Standing is determined by:
        # 1. Shoulders much higher than hips (large negative distance)
        # 2. Hips higher than knees (negative distance)
        # 3. Shoulders in upper half of frame
        if (shoulder_hip_distance > 0.1 and  # Significant vertical gap between shoulders and hips
            shoulder_height < 0.45):         # Shoulders in upper part of frame
            return "standing"
        
        # Sitting is when shoulders are above hips but closer together vertically
        # and overall position is lower in frame
        else:
            return "sitting"

    def _detect_drinking(self, frame, pose_results):
        """Enhanced drinking detection for bottles, glasses, and cups"""
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with hand detection
        hand_results = self.hands.process(rgb_frame)
        
        # If no pose landmarks are detected, can't determine drinking
        if not pose_results.pose_landmarks:
            return False
            
        # Get pose landmarks
        landmarks = pose_results.pose_landmarks.landmark
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        mouth_left = landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT]
        
        # Calculate distances and angles
        mouth_y = (mouth_left.y + mouth_right.y) / 2
        mouth_x = (mouth_left.x + mouth_right.x) / 2
        left_hand_to_mouth = ((left_wrist.x - mouth_x)**2 + (left_wrist.y - mouth_y)**2)**0.5
        right_hand_to_mouth = ((right_wrist.x - mouth_x)**2 + (right_wrist.y - mouth_y)**2)**0.5
        
        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Object detection for drinking containers
        height, width = frame.shape[:2]
        left_container = False
        right_container = False
        
        # Check for objects near hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Use middle finger tip as reference point
                middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
                hand_x, hand_y = int(middle_tip.x * width), int(middle_tip.y * height)
                
                # Extract region around hand
                hand_region_size = min(150, min(width, height) // 3)  # Larger region to capture glasses
                x1 = max(0, hand_x - hand_region_size)
                y1 = max(0, hand_y - hand_region_size)
                x2 = min(width, hand_x + hand_region_size)
                y2 = min(height, hand_y + hand_region_size)
                
                hand_region = frame[y1:y2, x1:x2]
                if hand_region.size > 0:
                    # Check if this region contains a drinking container
                    has_container = self._detect_drinking_container(hand_region)
                    
                    # Determine if it's left or right hand
                    if middle_tip.x < 0.5:  # Left side of image
                        left_container = has_container
                    else:  # Right side of image
                        right_container = has_container
        
        current_time = time.time()
        if current_time - self.last_drink_detection > 2:  # 2-second cooldown
            # Drinking is detected when:
            # 1. Hand is very close to mouth (< 0.15 distance)
            # 2. Elbow is bent (angle < 140 degrees)
            # 3. Container detected in hand region (if hands detected)
            left_drinking = left_hand_to_mouth < 0.15 and left_elbow_angle < 140
            right_drinking = right_hand_to_mouth < 0.15 and right_elbow_angle < 140
            
            # If hands are very close to mouth, it's likely drinking even without object detection
            very_close_to_mouth = min(left_hand_to_mouth, right_hand_to_mouth) < 0.08
            
            # Check if conditions are met
            if very_close_to_mouth or \
               (left_drinking and (not hand_results.multi_hand_landmarks or left_container)) or \
               (right_drinking and (not hand_results.multi_hand_landmarks or right_container)):
                self.last_drink_detection = current_time
                print("Debug - Drinking detected!")
                print(f"Debug - Hand distances: L={left_hand_to_mouth:.3f}, R={right_hand_to_mouth:.3f}")
                print(f"Debug - Elbow angles: L={left_elbow_angle:.1f}, R={right_elbow_angle:.1f}")
                print(f"Debug - Container detected: L={left_container}, R={right_container}")
                return True
        
        return False

    def _detect_drinking_container(self, region):
        """Detect bottles, glasses, and cups in the region"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for container shapes
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check for containers based on shape
            aspect_ratio = h / w if w > 0 else 0
            
            # Check for different container types:
            # 1. Bottles (taller than wide)
            if aspect_ratio > 1.5:
                return True
                
            # 2. Glasses/cups (roughly square or slightly taller than wide)
            if 0.7 < aspect_ratio < 1.5:
                # Check if contour is somewhat round/curved (for glass/cup detection)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # If the shape has few corners or is rounded, likely a glass/cup
                if len(approx) < 10:
                    # Area vs perimeter check - circles/ovals have higher ratio
                    area = cv2.contourArea(contour)
                    area_perimeter_ratio = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Higher ratio means more circular/oval (cups, glasses)
                    if area_perimeter_ratio > 0.5:
                        return True
        
        return False

    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (in degrees)"""
        vector1 = [point1.x - point2.x, point1.y - point2.y]
        vector2 = [point3.x - point2.x, point3.y - point2.y]
        
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = (vector1[0]**2 + vector1[1]**2)**0.5
        magnitude2 = (vector2[0]**2 + vector2[1]**2)**0.5
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def process_video_segment(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Process a segment of video frames and return combined pose and model-based predictions."""
        if not frames or len(frames) < 8:
            return {}
        # Pose detection
        pose_preds = []
        for frame in frames[:16]:
            p = self._analyze_pose(frame)
            if p:
                pose_preds.append(p)
        # Early fall detection
        if 'falling' in pose_preds:
            return {'falling': 0.95}
        pose_scores = {}
        if pose_preds:
            counts = Counter(pose_preds)
            total = len(pose_preds)
            for act, cnt in counts.items():
                pose_scores[act] = cnt / total * 0.9

        # Model predictions
        processed = [self.preprocess_frame(f) for f in frames[:16]]
        processed = [f for f in processed if f is not None]
        if len(processed) < 8 or self.model is None:
            candidates = {a: s for a, s in pose_scores.items() if s > self.confidence_threshold}
            return dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:1])
        clip = torch.stack(processed).permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            try:
                logits = self.model(clip)[0]
                probs = F.softmax(logits, dim=0).cpu().numpy()
            except Exception as e:
                print(f"Error running model: {e}")
                probs = np.zeros(len(self.class_names))
        model_scores = {}
        for label, idxs in self.class_indices_map.items():
            model_scores[label] = float(max(probs[idxs])) if idxs else 0.0

        combined = {l: max(pose_scores.get(l, 0.0), model_scores.get(l, 0.0)) for l in self.simple_action_map}
        results = {a: c for a, c in combined.items() if c > self.confidence_threshold}
        out = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:1])
        if out:
            for a, v in out.items():
                print(f"Detected {a}: {v*100:.1f}%")
        return out
    
    def is_suspicious_activity(self, predictions: Dict[str, float], 
                             suspicious_actions: List[str], 
                             threshold: float = 0.5) -> bool:
        """Determine if the current activity is suspicious"""
        try:
            for action, confidence in predictions.items():
                if action in suspicious_actions and confidence > threshold:
                    return True
            return False
        except Exception as e:
            print(f"Error checking suspicious activity: {e}")
            return False

    def _detect_looking_at_phone(self, frame, landmarks):
        """Detect if person is looking at phone based on hand position and object detection"""
        # Extract frame dimensions
        height, width = frame.shape[:2]
        
        # Get relevant landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # Calculate positions in pixel coordinates
        nose_pos = (int(nose.x * width), int(nose.y * height))
        left_wrist_pos = (int(left_wrist.x * width), int(left_wrist.y * height))
        right_wrist_pos = (int(right_wrist.x * width), int(right_wrist.y * height))
        
        # Check if either hand is in front of face (between face and camera)
        # Phone is typically held in front of face when looking at it
        left_hand_x_distance = abs(left_wrist.x - nose.x)
        right_hand_x_distance = abs(right_wrist.x - nose.x)
        
        # Criteria:
        # 1. Hand is close to face horizontally (in front of face)
        # 2. Hand is somewhere between chest and face vertically
        # 3. Elbow is bent, indicating holding something
        left_elbow_angle = self._calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER], 
                                                 left_elbow, 
                                                 left_wrist)
        right_elbow_angle = self._calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], 
                                                  right_elbow, 
                                                  right_wrist)
        
        # Check for phone-like object in region around hands
        left_phone_detected = False
        right_phone_detected = False
        
        # Define search regions around wrists
        wrist_region_size = min(150, min(width, height) // 3)
        
        # Left hand region
        left_x1 = max(0, left_wrist_pos[0] - wrist_region_size//2)
        left_y1 = max(0, left_wrist_pos[1] - wrist_region_size//2)
        left_x2 = min(width, left_wrist_pos[0] + wrist_region_size//2)
        left_y2 = min(height, left_wrist_pos[1] + wrist_region_size//2)
        
        # Right hand region
        right_x1 = max(0, right_wrist_pos[0] - wrist_region_size//2)
        right_y1 = max(0, right_wrist_pos[1] - wrist_region_size//2)
        right_x2 = min(width, right_wrist_pos[0] + wrist_region_size//2)
        right_y2 = min(height, right_wrist_pos[1] + wrist_region_size//2)
        
        # Check for phone in left hand region
        if left_x2 > left_x1 and left_y2 > left_y1:
            left_hand_region = frame[left_y1:left_y2, left_x1:left_x2]
            if left_hand_region.size > 0:
                left_phone_detected = self._detect_phone_in_region(left_hand_region)
        
        # Check for phone in right hand region
        if right_x2 > right_x1 and right_y2 > right_y1:
            right_hand_region = frame[right_y1:right_y2, right_x1:right_x2]
            if right_hand_region.size > 0:
                right_phone_detected = self._detect_phone_in_region(right_hand_region)
        
        # Decision criteria for looking at phone:
        # 1. Hand is relatively close to face horizontally
        # 2. Elbow is bent (< 140 degrees)
        # 3. Phone-like object detected in hand region
        left_criteria = (left_hand_x_distance < 0.2 and 
                         left_elbow_angle < 140 and 
                         (left_phone_detected or left_wrist.y < nose.y))
        
        right_criteria = (right_hand_x_distance < 0.2 and 
                          right_elbow_angle < 140 and 
                          (right_phone_detected or right_wrist.y < nose.y))
        
        if left_criteria or right_criteria:
            print("Debug - Looking at phone detected!")
            if left_phone_detected:
                print("Debug - Phone detected in left hand")
            if right_phone_detected:
                print("Debug - Phone detected in right hand")
            return True
            
        return False
        
    def _detect_phone_in_region(self, region):
        """Detect phone-like objects in the region"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for rectangular objects (phones)
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 200:  # Phones should have decent area
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = h / w if w > 0 else 0
            
            # Phone criteria:
            # 1. Rectangular shape (aspect ratio between 1.5 and 2.5)
            # 2. Fairly large area
            # 3. Not too complex shape (simple rectangular outline)
            if 1.5 < aspect_ratio < 2.5 or 0.4 < aspect_ratio < 0.7:
                # Check if contour is fairly rectangular
                rect_area = w * h
                contour_area = cv2.contourArea(contour)
                area_ratio = contour_area / rect_area if rect_area > 0 else 0
                
                # If area ratio is high (close to rectangle) and significant size
                if area_ratio > 0.7 and contour_area > 1000:
                    return True
                    
            # Alternative: Check for rounded rectangle (for newer phones)
            if 1.7 < aspect_ratio < 2.2:
                # Approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # Phones typically have 4-8 corners after approximation
                if 4 <= len(approx) <= 8:
                    return True
        
        return False

    def _detect_fall_position(self, landmarks, frame_height, frame_width):
        """
        Detect if the person is in a falling position based on body orientation and position
        This detects the actual position of falling, not just the movement
        """
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Calculate key positions
        shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_center = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        ankle_center = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)
        
        # Calculate body angle (angle between vertical and the line from hip to shoulder)
        # When standing, this angle should be close to 0
        # When falling or lying, this angle will be much larger
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        body_angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        # Calculate leg angle (angle between vertical and the line from hip to ankle)
        leg_dx = ankle_center[0] - hip_center[0]
        leg_dy = ankle_center[1] - hip_center[1]
        leg_angle = abs(np.degrees(np.arctan2(leg_dx, leg_dy)))
        
        # Check for abnormal body orientation (tilted far from vertical)
        is_body_horizontal = body_angle > 45  # Body tilted more than 45 degrees
        
        # Check for body parts at similar heights (indicates horizontal position)
        shoulder_height = shoulder_center[1]
        hip_height = hip_center[1]
        height_difference = abs(shoulder_height - hip_height)
        
        # Check for legs being up or in abnormal position
        leg_position_abnormal = leg_angle > 60
        
        # Check for overall body position in frame
        hip_is_high = hip_center[1] < 0.5  # Hip position in upper half of frame
        
        # Calculate if the body is in an unbalanced position
        knee_height = (left_knee.y + right_knee.y) / 2
        ankle_height = (left_ankle.y + right_ankle.y) / 2
        legs_above_body = knee_height < hip_height or ankle_height < hip_height
        
        # Debug information
        print(f"Debug - Fall detection: Body angle: {body_angle:.1f}, Leg angle: {leg_angle:.1f}")
        print(f"Debug - Height difference: {height_difference:.3f}, Hip height: {hip_center[1]:.2f}")
        
        # Combined criteria for fall detection:
        # 1. Body is significantly tilted (> 45 degrees)
        # 2. Either:
        #    a. Shoulders and hips are at similar heights (horizontal orientation)
        #    b. Legs are in abnormal position (raised or twisted)
        #    c. Body parts are in unusual relative positions
        current_time = time.time()
        
        # Check if sufficient time has passed since last fall detection
        if current_time - self.last_fall_detection <= 5:
            return False
            
        if (is_body_horizontal and 
            (height_difference < 0.15 or leg_position_abnormal or legs_above_body)):
            self.last_fall_detection = current_time
            print("FALL POSITION DETECTED!")
            print(f"Body angle: {body_angle:.1f}, Legs abnormal: {leg_position_abnormal}")
            return True
            
        return False 