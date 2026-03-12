from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# CONFIGURACION GENERAL
# =========================
VIDEO_PATH = "videos/personas.mp4"
OUTPUT_VIDEO = "salidas/resultado_tracking_heatmap.mp4"
OUTPUT_HEATMAP = "salidas/heatmap_final.png"

MODEL_PATH = "yolo26n.pt"       # Puede cambiarse por otro peso compatible
TRACKER = "botsort.yaml"        # Cambie a "bytetrack.yaml" para comparar
CLASSES = [0]                   # Persona
CONF = 0.35
IOU = 0.50
MAX_HISTORY = 40                # Longitud maxima de la trayectoria por ID
HEAT_RADIUS = 25                # Radio de acumulacion en el mapa de calor
ALPHA = 0.55                    # Transparencia de heatmap sobre la imagen


def ensure_output_dir():
    Path("salidas").mkdir(parents=True, exist_ok=True)


def draw_track_history(frame, history_dict):
    """Dibuja rutas historicas por ID."""
    for track_id, points in history_dict.items():
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(230, 230, 230), thickness=3)
        cv2.putText(
            frame,
            f"ID {track_id}",
            tuple(points[-1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def add_to_heatmap(heatmap, center, radius=25):
    """Acumula energia en el mapa de calor alrededor del centro."""
    x, y = center
    h, w = heatmap.shape[:2]

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)

    if x1 >= x2 or y1 >= y2:
        return

    roi = heatmap[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    gaussian = np.exp(-dist_sq / (2 * (radius / 2.5) ** 2)) * 10.0

    heatmap[y1:y2, x1:x2] = roi + gaussian.astype(np.float32)


def create_heat_overlay(frame, heatmap):
    """Convierte la matriz acumulada en overlay de color."""
    normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - ALPHA, colored, ALPHA, 0)
    return overlay, colored


def main():
    ensure_output_dir()

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    track_history = defaultdict(list)
    heatmap = np.zeros((height, width), dtype=np.float32)

    while True:
        success, frame = cap.read()
        if not success:
            break

        result = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            classes=CLASSES,
            conf=CONF,
            iou=IOU,
            verbose=False,
        )[0]

        annotated = result.plot()

        if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box

                # Centro del bounding box
                cx, cy = int(x), int(y)

                # Tambien puede usar "pie" de la persona:
                # cx, cy = int(x), int(y + h / 2)

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > MAX_HISTORY:
                    track_history[track_id].pop(0)

                add_to_heatmap(heatmap, (cx, cy), radius=HEAT_RADIUS)

                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        draw_track_history(annotated, track_history)
        blended, colored_heat = create_heat_overlay(annotated, heatmap)

        writer.write(blended)
        cv2.imshow("Tracking + Rutas + Heatmap", blended)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.imwrite(OUTPUT_HEATMAP, colored_heat)
    cv2.destroyAllWindows()

    print("Proceso finalizado.")
    print(f"Video guardado en: {OUTPUT_VIDEO}")
    print(f"Mapa de calor final guardado en: {OUTPUT_HEATMAP}")


if __name__ == "__main__":
    main()
