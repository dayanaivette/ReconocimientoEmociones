import cv2
import os
import numpy as np
import time

directorio_entrenamiento = "C:/Users/dayan/Desktop/ReconocimientoEmociones"
nuevo_ancho = 150
nuevo_alto = 150

def obtenerModelo(method,facesData, labels):
    
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces':emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'LBPH':emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    
    print("Entrenando ( "+method+" )...")
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time() - inicio
    print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)
    emotion_recognizer.write("modelo"+method+".xml")
emotionsList = os.listdir(directorio_entrenamiento)
print('Lista de emociones:', emotionsList)

labels = []
facesData = []
label = 0

for emotion in emotionsList:
    emotionPath = os.path.join(directorio_entrenamiento, emotion)
    
    if not os.path.isdir(emotionPath):  # Verificar si es un directorio
        continue
    
    for filename in os.listdir(emotionPath):
        imagePath = os.path.join(emotionPath, filename)
        
        if not os.path.isfile(imagePath):  # Verificar si es un archivo
            continue
        
        image = cv2.imread(imagePath)
        
        if image is None:
            print("No se pudo cargar la imagen:", imagePath)
            continue
        
        image_resized = cv2.resize(image, (nuevo_ancho, nuevo_alto))
        
        facesData.append(image_resized)
        labels.append(label)   
    label += 1
facesData = np.array(facesData)

obtenerModelo('EigenFaces',facesData, labels)
obtenerModelo('FisherFaces',facesData,labels)
obtenerModelo('LBPH',facesData,labels)