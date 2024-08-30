import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras import backend as K

#För att undvika att modellenn laddas flera ggr
if'classifier' not in st.session_state:
    K.clear_session()
    st.session_state.classifier = load_model('model-2024-08-30.keras')

classifier = st.session_state.classifier    

# Ladda din tränade modell
face_classifier = cv2.CascadeClassifier(r"C:\Users\ASUS\OneDrive\Skrivbord\AI2\haarcascade_frontalface_default.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Funktion för att förutsäga ansiktsuttryck från en bild
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    accuracy = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        accuracy = max(prediction)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, label, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(image, f"{accuracy:.2%}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
       
    return image, accuracy


# Anpassad titel med färg
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>😄 Ansiktsuttryck 😨</h1>", unsafe_allow_html=True)

# Livestream-funktionalitet
if'run' not in st.session_state:
    st.session_state.run = False

def start_stream():
    st.session_state.run = True

def stop_stream():
    st.session_state.run = False

st.button('Starta Livestream', on_click=start_stream)
st.button('Stoppa Livestream', on_click=stop_stream)


if st.session_state.run:
    camera = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while st.session_state.run:
        _, frame = camera.read()
        frame, accuracy = predict_emotion(frame)
        FRAME_WINDOW.image(frame, channels='BGR')

    camera.release()
    cv2.destroyAllWindows()
else:   
    #Anpassad titel och färg
    st.markdown("<h1 style='text-align: center; color: #FF6347;'>Livestream pausad</h3>", unsafe_allow_html=True)   

# Uppladdning av bild
uploaded_file = st.file_uploader("Välj en bild...", type="jpg")

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)

    # Visa uppladdad bild
    st.image(image, channels="BGR")

   # Förutsäg ansiktsuttryck
    image, accuracy = predict_emotion(image)
    st.image(image, channels="BGR", caption="Avläst ansiktsuttryck")

    
    
    