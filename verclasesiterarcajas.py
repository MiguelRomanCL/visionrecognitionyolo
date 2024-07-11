# %%
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# Cargar el modelo
model = YOLO("yolov8x.pt")  # Asegúrate de que la ruta al modelo es correcta

# Ruta de la imagen a analizar
image_path = "./inputimages/fotocumple.jpeg"


# Realizar la predicción
results = model.predict(image_path)

# Crear una carpeta para guardar las predicciones si no existe
output_folder = "./detect"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Procesar cada resultado (para una imagen, generalmente solo hay uno)
for i, result in enumerate(results):
    # Crear una subcarpeta para cada predicción
    prediction_folder = os.path.join(output_folder, f"predict{i+1}")
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    # Guardar la imagen con las cajas detectadas
    result.save(prediction_folder)

    # Visualizar la imagen con las cajas detectadas
    plt.imshow(result.plot(show=False))
    plt.axis("off")
    plt.savefig(os.path.join(prediction_folder, "prediction.png"))
    plt.close()

# Imprimir todas las cajas detectadas
print("Cajas detectadas:")
for result in results:
    names = result.names
    for box in result.boxes:
        box_coords = box.xyxy.tolist()
        box_conf = box.conf[0].item()
        box_class = int(box.cls[0].item())
        print(
            f"Coordenadas: {box_coords}, Confianza: {box_conf}, Clase: {names[box_class]}"
        )


# %%
from ultralytics import YOLO

# Cargar el modelo
model = YOLO("yolov8x")

# Ruta de la imagen a analizar
image_path = "./inputimages/imagen2pollos.jpeg"
image_path = "./inputimages/fotocumple.jpeg"

# Realizar la predicción
result = model.predict(image_path)[0]  # Solo tomamos el primer resultado

# Obtener los nombres de las clases del modelo
names = result.names

# Imprimir todas las cajas detectadas
print("Cajas detectadas:")
for box in result.boxes:
    box_coords = box.xyxy.tolist()
    box_conf = box.conf[0].item()
    box_class = int(box.cls[0].item())
    print(
        f"Coordenadas: {box_coords}, Confianza: {box_conf}, Clase: {names[box_class]}"
    )


# %%

# Realizar la predicción
result = model.predict(image_path)[0]  # Solo tomamos el primer resultado

# Lista de clases relacionadas con pollos o aves
chicken_related_classes = [
    "bird"
]  # Puedes ajustar esto según lo que consideres relevante

# Obtener los nombres de las clases del modelo
names = result.names

# Filtrar cajas por clases relacionadas con pollos o aves y confianza > 0.6
filtered_boxes = [
    box
    for box in result.boxes
    if names[int(box.cls[0].item())] in chicken_related_classes
    and box.conf[0].item() > 0.6
]

# Crear las cajas filtradas manualmente
filtered_boxes_data = [
    [box.xyxy.tolist(), box.conf[0].item(), int(box.cls[0].item())]
    for box in filtered_boxes
]

# Imprimir los resultados filtrados
print("Cajas filtradas:")
for box_data in filtered_boxes_data:
    print(
        f"Coordenadas: {box_data[0]}, Confianza: {box_data[1]}, Clase: {names[box_data[2]]}"
    )


# %%
result.boxes

# %%
{
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# %%

# Realizar la predicción
result = model.predict(image_path)[0]  # Solo tomamos el primer resultado

# Lista de clases relacionadas con pollos o aves
chicken_related_classes = [
    "bird"
]  # Puedes ajustar esto según lo que consideres relevante

# Obtener los nombres de las clases del modelo
names = result.names

# Filtrar cajas por clases relacionadas con pollos o aves y confianza > 0.6
filtered_boxes = [
    box
    for box in result.boxes
    if names[int(box.cls[0].item())] in chicken_related_classes
    and box.conf[0].item() > 0.6
]

# Crear las cajas filtradas manualmente
filtered_boxes_data = [
    [box.xyxy.tolist(), box.conf[0].item(), int(box.cls[0].item())]
    for box in filtered_boxes
]

# Imprimir los resultados filtrados
print("Cajas filtradas:")
for box_data in filtered_boxes_data:
    print(
        f"Coordenadas: {box_data[0]}, Confianza: {box_data[1]}, Clase: {names[box_data[2]]}"
    )


# %%
filtered_boxes

# %%

# Cargar el modelo
model = YOLO("yolov8x")

# Ruta de la imagen a analizar
image_path = "./inputimages/imagen1pollos.jpeg"

# Realizar la predicción
results = model.predict(image_path)

# Procesar cada resultado en la lista de resultados
result = results[0]
# Obtener los nombres de las clases del modelo
names = result.names

# Filtrar cajas por la clase 'person' y confianza > 0.6
filtered_boxes = [
    box
    for box in result.boxes
    if names[int(box.cls[0].item())] == "person" and box.conf[0].item() > 0.6
]

# Crear un nuevo objeto Results con las cajas filtradas
filtered_result = Results(boxes=Boxes(objects=filtered_boxes), names=names)

# Aquí puedes añadir el código para guardar o mostrar los resultados filtrados
# por ejemplo:
print(filtered_result)


# %%
len(results)

# %%
from ultralytics import YOLO

# Cargar el modelo
model = YOLO("yolov8x")

# Ruta de la imagen a analizar
# image_path = './inputimages/fotocumple.jpeg'
image_path = "./inputimages/imagen1pollos.jpeg"

# Realizar la predicción
results = model.predict(image_path)

result = results[0]

# Obtener los nombres de las clases del modelo
names = result.names

# Filtrar cajas por la clase 'person' y confianza > 0.6
filtered_boxes = [
    box
    for box in result.boxes
    if names[int(box.cls[0].item())] == "person" and box.conf[0].item() > 0.6
]

# Crear un nuevo objeto Results con las cajas filtradas (si la API lo permite)
filtered_result = ultralytics.engine.results.Results(
    boxes=ultralytics.engine.results.Boxes(objects=filtered_boxes), names=names
)


# %%
result

# %%
# Guardar el resultado filtrado
# Este paso depende de la capacidad de la API de YOLO para guardar directamente un objeto Results modificado
# Si la API no lo permite, tendrás que usar un método alternativo como el ejemplo de PIL mostrado anteriormente
model.save(filtered_result, "./outputimages/filtradas_fotocumple.jpeg")


# %%


# %%


# %%
from ultralytics import YOLO

model = YOLO("yolov8x")

image_path = "./inputimages/fotocumple.jpeg"

result = model.predict(image_path, save=True)

current_frame = result[0]

for box in current_frame.boxes:
    print(box)


box = current_frame.boxes[0]

# %%
classes_of_interest = ["person", "bicycle", "car"]

# Filtramos las cajas para mantener sólo las que pertenecen a las clases de interés
filtered_boxes = [
    box
    for box in current_frame.boxes
    if names[int(box.cls[0].item())] in classes_of_interest
]
filtered_boxes

# %%
# Asumimos que tienes 'filtered_boxes' como se definió anteriormente
# Crear un nuevo objeto Results que solo contenga las cajas filtradas
filtered_result = ultralytics.engine.results.Results(
    boxes=ultralytics.engine.results.Boxes(objects=filtered_boxes), names=result.names
)

# Guardar la imagen con los cuadros de delimitación filtrados
# La función save() o algo similar debería ser proporcionada por Ultralytics
# Esto dependerá de cómo la API permita guardar resultados modificados
model.save(filtered_result, "./outputimages/filtradas_fotocumple.jpeg")


# %%
names = result[0].names

# %%
names

# %%
print(result)
