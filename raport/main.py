from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from time import sleep

script_dir = Path(__file__).parent.resolve()
#Set file paths
model_file = script_dir/"models/model_edgetpu2.tflite"
data_dir = script_dir/"data"
label_file = script_dir/"labels.txt"

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()
#Jump through each photo
for x in range(1 , 207):
   #Set image path and resize it to be compatible with coral
   image_file =str(data_dir)+"/image"+str(x)+'.jpg' 
   size = common.input_size(interpreter)
   image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
   img = Image.open(image_file)

   common.set_input(interpreter, image)
   interpreter.invoke()
   classes = classify.get_classes(interpreter, top_k=7)


   y=10
   #Set font and labels path
   myfont = ImageFont.truetype(str(script_dir)+'/arial.ttf', 20)
   labels = read_label_file(label_file)
   #Jump through each cloud type
   for c in classes:
      #Assign cloud type to photo if detected
      print("image" +str(x)+" "+f'{labels.get(c.id, c.id)} {c.score:.5f}') 
      draw = ImageDraw.Draw(img)
      bbox = draw.textbbox((10,y), f'{labels.get(c.id, c.id)} {c.score:.5f}', font = myfont)
      draw.rectangle(bbox, fill="black")
      i1 = ImageDraw.Draw(img)
      i1.text((10, y),f'{labels.get(c.id, c.id)} {c.score:.5f}', fill = (255,255,255),font = myfont)
      y+=25
   #Save image
   img.save(str(script_dir)+"/results/image"+str(x)+'.jpg' )