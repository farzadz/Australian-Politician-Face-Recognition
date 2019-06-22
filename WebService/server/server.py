from flask import Flask, jsonify
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
from flask import request
from facedetector import extractor
from recognizer import classifier
import cv2

app = Flask(__name__)

face_extractor = extractor.FaceExtractor()

@app.route('/face_recognition/api/v1.0/', methods=['POST'])
def get_tasks():

    if not request.json or not 'pic_string' in request.json or request.json['format'] != 'JPG':
        abort(400)
    
    decoded = base64.b64decode(request.json['pic_string'])
    image = mpimg.imread(io.BytesIO(decoded), format='JPG')
    
    faces = face_extractor.extract_faces(image)
    
    response = {'num_faces_detected': len(faces), 'most_likely_politicians': []}
    for i, face in enumerate(faces):
        
#         byte_array = io.BytesIO()
#         face.save(byte_array, format='JPG')
#         byte_array = byte_array.getvalue()
        
        name = classifier.predict_label(face)
        
        #base64 encode image and add to response json
        
        
        # tricking cv2
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode('.jpg', face)
        face_bytes = encoded_image.tobytes()
        response['most_likely_politicians'].append({'base64_face': base64.b64encode(face_bytes).decode('utf-8'),
                                                    'predicted_name': name})
#         print(type(face_bytes), 'face_bytes type')
#         print('*' * 10)
        plt.imsave('{}.jpg'.format(i) , face)
    
    
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
