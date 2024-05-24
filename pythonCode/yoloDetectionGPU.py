import cv2  # Importer OpenCV for videobehandling
from ultralytics import YOLO  # Importer YOLO-modellen fra ultralytics-biblioteket
import torch  # Importer PyTorch for å jobbe med YOLO-modellen
import time  # Importer tid-modulen for å måle tid
import gpustat  # Importer gpustat for å overvåke GPU-bruk

# laster inn ferdigtrent YOLO-modell
model = YOLO('yolov8n.pt')

# path til video som brukes
vidPath = "" # legg til path her for input

# åpner videoen(vidPath) eller bruk (0) for camera
videoCapture = cv2.VideoCapture(0) # bruk 1 for koblet til kamera, eller vidPath for input

# ser om det er noe å vise
if not videoCapture.isOpened():
    print("Feil ved åpning av videofil.")
    exit()

# definerer variabler som skal brukes til å regne gjennomsnittet
totalInferenceTime = 0
totalFPS = 0
frameCount = 0


# Loop gjennom videoframes
while videoCapture.isOpened():
    # leser frames (bilder) fra en video eller kamera og behandler dem en etter en
    
    # ableToReadFrame vil være true hvis det er noe å ''lese'' og false ellers
    ableToReadFrames, frame = videoCapture.read()  # frame er bildet som blir lest fra videoen eller kameraet
    
    if ableToReadFrames:
        
        startInferenceTime = time.perf_counter()  # begynner å måler tid for inferanse (deteksjon)
        
        detectObjects = model(frame)  # gjør selve deteksjonen også kalt inferanse
        endInferenceTime = time.perf_counter()  # stopper tid for deteksjonen
        
        inferenceTime = endInferenceTime - startInferenceTime  # regner ut tiden brukt til inferanse (deteksjon)
        totalInferenceTime += inferenceTime  # legger tiden i total inferansetid
        
        
        markedFrame = detectObjects[0].plot()  # frames som har detekterte objekter blir markert

        # Viser FPS på bildet
        fps = 1 / inferenceTime  # regner ut FPS jo lavere prosesseringstid jo bedre fps
        totalFPS += fps  # legger til FPS-verdien for hver ramme
        cv2.putText(markedFrame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # skriver FPS på bildet fra ChatGPT

        # viser resultatet i eget vindu
        cv2.imshow("YOLOv8 Inference", markedFrame) 


        
       
        # Skriv ut prosesseringstid mens deteksjonen skjer
        print(f"Prosesseringstid (inferanse) for ramme {frameCount}: {inferenceTime:.4f} sekunder")
        

        frameCount += 1  # en tracker som holder oversikt over antall rammer mens deteksjonen kjører

        # 
        key = cv2.waitKey(1)
        if key == 27:  # hvis 'esc' blir trykket, så avslutter det
            break

# regner ut gjennomsnitt
if frameCount > 0:
    avgInferenceTime = totalInferenceTime / frameCount  # Gjennomsnittlig inferansetid
    avgFPS = totalFPS / frameCount  # Gjennomsnittlig FPS
    print(" ")
    print(f"Gjennomsnittlig prosesseringstid for alle rammer: {avgInferenceTime:.4f} sekunder")
    print(f"Gjennomsnittlig FPS: {avgFPS:.0f}")
else:
    print("Ingen rammer lest inn.")

# bruker gpustat for å få GPU informasjon
gpuStats = gpustat.GPUStatCollection.new_query()

# skriver ut GPU ytelse
for gpu in gpuStats.gpus:
    print(f"GPU {gpu.index}: {gpu.name}, GPU-ytelse: {gpu.utilization}%")

videoCapture.release()
cv2.destroyAllWindows()
