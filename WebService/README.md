## Face detection and Recognition
Dockerized version of Australian Politicians Face Recognition. The classification architecture is VGG-16 and face detection is done via MTCNN.

### Quick Start

You can try the performance of the model using the provided client notebook. For setting up the server:

`docker run -p 5000:5000 farzadz/vggface:1.4`

Sending requests can be done using curl, javascript or any other method that can send/recieve images via HTTP POST requests. The accepted image format is base64 encoded JPG. `client.ipynb` illustrates a sample workflow.
