import cv2
import numpy as np
from fer import FER
import time

# Charger le détecteur FER avec MTCNN
try:
    detector = FER(mtcnn=True)
    print("Modèle FER chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# Initialiser la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Boucle pour l'analyse en temps réel
try:
    while True:
        # Capturer une frame depuis la webcam
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer la frame.")
            break

        # Convertir en RGB pour FER
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détecter les émotions
        result = detector.detect_emotions(frame_rgb)

        # Traiter les résultats
        if result:
            for face in result:
                # Coordonnées du visage
                (x, y, w, h) = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Landmarks (caractéristiques faciales)
                landmarks = face.get('keypoints', {})
                if landmarks:
                    cv2.circle(frame, landmarks.get('left_eye', (0, 0)), 2, (0, 0, 255), -1)
                    cv2.circle(frame, landmarks.get('right_eye', (0, 0)), 2, (0, 0, 255), -1)
                    cv2.circle(frame, landmarks.get('mouth_left', (0, 0)), 2, (0, 0, 255), -1)
                    cv2.circle(frame, landmarks.get('mouth_right', (0, 0)), 2, (0, 0, 255), -1)

                # Émotion dominante
                emotions = face['emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                score = emotions[dominant_emotion]
                label = f"{dominant_emotion} ({score:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Afficher la frame avec les annotations
        cv2.imshow('Emotion Detection', frame)

        # Contrôler la vitesse (optionnel, ajustable)
        time.sleep(0.05)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nArrêt manuel.")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
print("Programme terminé.")