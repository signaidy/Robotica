Proyecto YOLO - Tracking + Heatmap

Estructura:
YOLO/
├── videos/
│   └── personas.mp4
├── salidas/
├── lab_tracking_heatmap.py
└── README.txt

Instalacion (desde el venv del repo):
- pip install ultralytics opencv-python numpy

Uso:
1) Coloque el video en: videos/personas.mp4
2) Coloque los pesos en la carpeta YOLO (por defecto: yolo26n.pt) o cambie MODEL_PATH
3) Ejecute: python lab_tracking_heatmap.py

Salidas:
- salidas/resultado_tracking_heatmap.mp4
- salidas/heatmap_final.png

Nota: Presione la tecla "q" para salir durante la reproduccion.
