from flask import Flask, request, redirect, url_for, jsonify
import face_recognition
import cv2
import numpy as np


app = Flask(__name__)

@app.route("/")
def index():
    return "HelloWorld"

@app.route("/face-compare", methods=['POST'])
def faceCompare():
    if request.method == 'POST':
        face1 = request.form.get('face1')
        face2 = request.form.get('face2')
        face2_bytes = open(face2, 'rb')
        np_array = np.frombuffer(face2_bytes.read(), np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < 100: 
            return {'error': 'Yuz sifati bilan muommo yuzaga keldi, iltimos kameraga yuzingizni qarating!'}
        # Load face images
        image1 = face_recognition.load_image_file(face1)
        image2 = face_recognition.load_image_file(face2)
        # Encode faces
        encoding1 = face_recognition.face_encodings(image1)
        encoding2 = face_recognition.face_encodings(image2)
        if not encoding2: 
            return jsonify({'error': 'No face found FACE2'})
        
        if not encoding1 :
            return jsonify({'error': 'No face found FACE1'})
        
        # Compare faces
        result = face_recognition.compare_faces([encoding1[0]], encoding2[0])

        # result is a list of True/False values indicating if the faces match
        if result[0]:
            return jsonify({'match': True, 'confidence': face_recognition.face_distance([encoding1[0]], encoding2[0])[0]})
        else:
            return jsonify({'match': False, 'confidence': None})

    return "HelloWorld"

if __name__ == "__main__":
    # Please do not set debug=True in production
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)