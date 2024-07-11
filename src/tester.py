# %%

from ultralytics import YOLO
import os


def predict_and_save(
    model_path,
    input_path,
    output_path=os.getenv("path_runs_yolo_predict"),
    filter_classes=None,
    save_value=True,
):
    # Cargar el modelo
    model = YOLO(model_path)

    # Realizar la predicción y guardar los resultados
    results = model.predict(input_path, save=save_value, project=output_path)

    # Obtener el primer resultado
    result = results[0]

    # Obtener los nombres de las clases del modelo
    names = result.names

    # Filtrar cajas si se proporciona un filtro de clases
    if filter_classes:
        filtered_boxes = [
            box
            for box in result.boxes
            if names[int(box.cls[0].item())] in filter_classes
        ]
    else:
        filtered_boxes = result.boxes

    # Imprimir el resultado completo
    print(results)

    print("***************************************")
    print("***************************************")
    print("Boxes")
    print("***************************************")
    print("***************************************")

    # Imprimir las cajas detectadas (filtradas o no)
    for box in filtered_boxes:
        box_coords = box.xyxy.tolist()
        box_conf = box.conf[0].item()
        box_class = int(box.cls[0].item())
        print(
            f"Coordenadas: {box_coords}, Confianza: {box_conf}, Clase: {names[box_class]}"
        )


# Ejemplo de uso
model_path = "yolov8s.pt"
input_path = "./inputimages/fotocumple.jpeg"
filter_classes = [
    "person",
    "chair",
]  # Ajusta esto según las clases que desees filtrar o usa None para no filtrar

predict_and_save(model_path, input_path, filter_classes=filter_classes, save_value=True)
