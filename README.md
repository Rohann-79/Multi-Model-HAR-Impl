# Intelligent Surveillance System

An advanced surveillance system using deep learning for real-time activity recognition and suspicious behavior detection. This system is designed for use in gated communities and elderly care facilities.

## Features

- Real-time activity recognition using state-of-the-art deep learning model (SlowR50)
- Pre-trained on Kinetics-400 dataset for accurate action detection
- Automatic detection of suspicious activities
- SMS alerts via Twilio integration
- Video recording of detected events
- Real-time visualization of detected activities
- Configurable alert system

## Project Structure

```
surveillance_project/
├── assets/               # Contains static files like alarm sounds
├── config/              # Configuration files
├── models/              # ML model and related files
│   ├── weights/         # Pre-trained model weights
│   └── action_recognition.py
├── recordings/          # Stores recorded event videos
├── requirements.txt     # Project dependencies
└── surveillance_system.py  # Main application
```

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained model:
```bash
mkdir -p models/weights
wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth -O models/weights/slow_r50_kinetics400.pth
```

4. Create necessary directories:
```bash
mkdir -p assets config recordings
```

5. Set up environment variables:
Create a `.env` file in the project root with:
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone
TARGET_PHONE_NUMBER=your_phone_number
```

6. Add an alarm sound file:
Place an alarm sound file (WAV format) in the `assets` folder named `alarm.wav`

## Running the System

Start the surveillance system:
```bash
python surveillance_system.py
```

## Configuration

The system can be configured through `config/surveillance_config.json`. Key settings include:

- Frame buffer size for activity detection
- Alert cooldown period
- Confidence threshold for alerts
- List of suspicious activities to monitor
- Recording settings
- Display options
- Notification preferences

## Technical Details

### Model Architecture
- Uses SlowR50 architecture from PyTorchVideo
- Pre-trained on Kinetics-400 dataset
- Capable of recognizing 400 different human activities
- Real-time processing with optimized inference

### Activity Detection
- Continuous video stream processing
- Frame buffering for temporal analysis
- Activity confidence scoring
- Configurable suspicious activity detection

### Alert System
- Multi-channel notifications (SMS, sound alerts)
- Event video recording
- Customizable alert thresholds
- Cooldown period to prevent alert spam

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Webcam or IP camera
- Internet connection for SMS alerts
- Twilio account for notifications

## License

MIT License 