from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

script_dir = path(__file__).parent.resolve()

model_file = script_dir/'models/AstroPi-model-v1.tflite'
data_dir = script_dir/'data'
label_file = data_dir/'models/AstroPi-model-v1.txt'
image_file = data_dir/'tests'/''

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

commom.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

labels = read_label_file(label_file)
for c in classes:
    print(f'{labels.get(c.id, c.id)} {c.score:.5f}')