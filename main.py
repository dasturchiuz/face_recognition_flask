from flask import Flask, request, redirect, url_for, jsonify
import face_recognition

app = Flask(__name__)

@app.route("/")
def index():
    return "HelloWorld"

@app.route("/face-compare", methods=['POST'])
def faceCompare():
    if request.method == 'POST':
        # FileStorage object wrapper
        if 'face1' not in request.files or 'face2' not in request.files:
            return jsonify({'error': 'Two face images are required.'}), 400

        # Load face images
        image1 = face_recognition.load_image_file(request.files['face1'])
        image2 = face_recognition.load_image_file(request.files['face2'])

        # Encode faces
        encoding1 = face_recognition.face_encodings(image1)
        encoding2 = face_recognition.face_encodings(image2)

        if not encoding1 or not encoding2:
            return jsonify({'error': 'No face found in one or both images.'}), 400

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
    app.run(host="0.0.0.0", port=5000, debug=True )