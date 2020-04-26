import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_input
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
detector = MTCNN()
print("loaded MTCnn")

def load_face_model():
    try:
        face_model = load_model("weights/face_recognition_model.h5")
    except OSError:
        face_model =  VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        face_model.save("weights/face_recognition_model.h5")
    return face_model

def extract_face(pixels, required_size=(224, 224)):
	# load image from file
	#pixels = pyplot.imread(filename)
	# create the detector, using default weights
	#detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
    
	x1, y1, width, height = results[0]['box']
	print(str(x1) + "  " + str(y1) + "  " + str(width)+" "  +str(height))
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


def get_embeddings(pixels,model):
	# extract faces
	faces = extract_face(pixels)
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	samples = np.expand_dims(samples,axis = 0)
  # perform prediction
	yhat = model.predict(samples)
	return yhat


def is_match(name, candidate,model,database, thresh=0.5):
    # calculate distance between embeddings
    known_embedding = database[name]
    candidate_embeddings = get_embeddings(candidate,model)
    score = cosine(known_embedding, candidate_embeddings)
    if score <= thresh:
        return '>face is a Match (%.3f <= %.3f)' % (score, thresh)
    else:
        return '>face is NOT a Match (%.3f > %.3f)' % (score, thresh)
    

def add_to_database(img,username,database,model):
    database[username] =  get_embeddings(img,model)
    return username + "ADDED IN DATABASE"

def check_in_database(img,username,database,model):
    return is_match(username,img,model,database)

def names_in_database(database):
    names = str(database.keys())
    return names