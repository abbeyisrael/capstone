from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_from_directory
import pyautogui
import mimetypes
from datetime import datetime
import os
import cv2

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('welcome.html')

@app.route('/upload_page')
def upload_page():
    # Render the uploads page
    return render_template('upload_page.html')

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/procedure', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image_url = url_for('static', filename='uploads/' + file.filename)
        return render_template('procedure.html', image_url=image_url)
    
    return "Invalid file type. Allowed types are: png, jpg, jpeg, gif"

# Folder to store screenshots
SCREENSHOT_FOLDER = 'static/screenshots'
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

# Global camera object to keep the feed running
camera = cv2.VideoCapture(0)

@app.route('/take-screenshot', methods=['GET'])
def take_screenshot():
    """Capture a single frame from the live camera feed."""
    if not camera.isOpened():
        return jsonify({"message": "Error: Unable to access the camera"}), 500

    try:
        # Read a single frame
        ret, frame = camera.read()
        if not ret:
            return jsonify({"message": "Error: Unable to capture image"}), 500

        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(SCREENSHOT_FOLDER, f'screenshot_{timestamp}.png')

        # Save the frame
        cv2.imwrite(filename, frame)

        return jsonify({"message": "Screenshot Taken!", "filename": filename})

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


def generate_camera_feed():
    """Generate live camera feed frames."""
    if not camera.isOpened():
        return b'Error: Unable to access the camera'

    while True:
        success, frame = camera.read()
        if not success:
            break
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera_feed')
def camera_feed():
    """Route to stream the live camera feed."""
    return Response(generate_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close-camera')
def close_camera():
    """Cleanup to release the camera on app shutdown."""
    camera.release()
    return jsonify({"message": "Camera released"})

# @app.route('/static/videos/<path:filename>')
# def serve_video(filename):
#     file_path = f"static/videos/{filename}"
#     mime_type, _ = mimetypes.guess_type(file_path)
#     return send_from_directory("static/videos", filename, mimetype=mime_type or "video/mp4")

@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    video_path = os.path.join("static/videos", filename)
    
    # Check if file exists
    if not os.path.exists(video_path):
        abort(404)  # Return 404 if the file is missing

    try:
        mime_type, _ = mimetypes.guess_type(video_path)  # ✅ Fix NameError
        return send_from_directory("static/videos", filename, mimetype=mime_type or "video/mp4")
    except Exception as e:
        print(f"Error serving video: {e}")  # Print error in terminal
        abort(500)  # Return 500 if something else fails



if __name__ == '__main__':
    app.run(debug=True)





#GET: message is sent, and the server returns the data
# #POST: used to send HTML form data to the server
# @app.route('/patient_selection', methods=['POST', 'GET'])
# def patient_selection():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         if username == VALID_USERNAME and password == VALID_PASSWORD:
#             patients = [
#                 {"id": 1, "name": "John Doe", "age": 45, "gender": "Male", "blood_group": "O"},
#                 {"id": 2, "name": "Jane Doe", "age": 65, "gender": "Female", "blood_group": "B"},
#             ]
#             return render_template('patient_selection.html', username=username, patients=patients)
#         else:
#             error = "Invalid username or password!"
#             return render_template('login.html', error=error)

#     return redirect(url_for('login'))


# @app.route('/patient/<int:patient_id>')
# def patient_details(patient_id):
#     # Simulate fetching data from a database or data structure
#     patients = [
#         {"id": 1, "name": "John Doe", "age": 45, "gender": "Male", "blood_group": "O"},
#         {"id": 2, "name": "Jane Doe", "age": 65, "gender": "Female", "blood_group": "B"},
#     ]
#     # Find the patient with the given ID
#     patient = next((p for p in patients if p["id"] == patient_id), None)
#     if patient:
#         return render_template('patient_details.html', patient=patient)
#     else:
#         return "Patient not found", 404

# @app.route('/procedure')
# def procedure():
#     return render_template('procedure.html')


# @app.route('/logout')
# def logout():
#     return redirect(url_for('login'))


