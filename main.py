from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from picamera import PiCamera
from picamera import Color
from time import sleep
import os 
from orbit import ISS
from skyfield.api import load
from pathlib import Path
from datetime import datetime, timedelta


#CHECKING DATE TO ESTABLISH HOW LONG THE PROGRAM SHOULD WORK
start_time = datetime.now()
now_time = datetime.now()



#CONVERTING LOCALISATION DATA TO ANGLES
def convert(angle):
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

#DEFINING FUNCTION TO CAPTURE PHOTOS WITH LOCALISATION IN METATAGS 
def capture(camera, image):
   
    south, exif_latitude = convert(location.latitude)
    west, exif_longitude = convert(location.longitude)

    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
   
    
    camera.capture(image)


#DEFINING ABSOLUTE PATH
script_dir = Path(__file__).parent.resolve()

camera = PiCamera()


#DEFINING MACHINE LEARNING MODELS 
time_model_file = script_dir/'models/time_model.tflite'
data_dir = script_dir/'data'
time_label_file = data_dir/'time_labels.txt'
cloud_model_file = script_dir/'models/clouds_model.tflite'
cloud_label_file = data_dir/'clouds_labels.txt'

time_interpreter = make_interpreter(f"{time_model_file}")
time_interpreter.allocate_tensors()

cloud_interpreter = make_interpreter(f"{cloud_model_file}")
cloud_interpreter.allocate_tensors()

camera.start_preview()




x = 1

#MAIN PROGRAM LOOP 
while(now_time < start_time + timedelta(minutes=179)):
    sleep(10)
    #GETTING ISS POSITION
    t = load.timescale().now()
    position = ISS.at(t)
    location = position.subpoint()


    south, latS = convert(location.latitude)
    west, longS= convert(location.longitude)

    #CONVERTING LOCALISATION SAVING METHOD

    lat = f'{-ISS.coordinates().latitude.degrees:.1f}' if south else f'{ISS.coordinates().latitude.degrees:.1f}'
    latref = "S" if south else "N"
    long = f'{-ISS.coordinates().longitude.degrees:.1f}' if west else f'{ISS.coordinates().longitude.degrees:.1f}'
    longref = "W" if west else "E"
    current_position = str(lat) + str(latref)+" "+ str(long) + str(longref)
    #CAPTURING PHOTOS
    camera.resolution = (1296,972)
    camera.annotate_background = Color('black')
    camera.annotate_foreground = Color('white')
    camera.annotate_text_size = 27
    #ADDING INFO OVERLAY TO PHOTOS
    camera.annotate_text = " Image "+str(x)+"; \n Time: "+str(datetime.now())+"; \n Location: "+str(current_position)
    capture(camera, f"{data_dir}/image%s.jpg" % x)
    print (location)
    image_file = str(data_dir)+"/image"+str(x)+'.jpg' 
    size = common.input_size(time_interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

    #CHECKING WITH DAY/NIGHT MODEL
    common.set_input(time_interpreter, image)
    time_interpreter.invoke()
    classes = classify.get_classes(time_interpreter, top_k=1)

    #LABELING
    labels = read_label_file(time_label_file) 
    for c in classes:
        print("image"+str(x)+" "+f'{labels.get(c.id, c.id)} {c.score:.5f}') #RETURNING OUTPUT 
        if c.id != 1:
            os.remove(str(data_dir)+"/image"+str(x)+'.jpg') #DELETING NIGHT AND TWILIGHT PHOTOS
        else:
            size = common.input_size(cloud_interpreter)
            image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
            #CHECKING WITH CLODUS MODEL
            common.set_input(cloud_interpreter, image)
            cloud_interpreter.invoke()
            classes = classify.get_classes(cloud_interpreter, top_k=1)

            #LABELING
            labels = read_label_file(cloud_label_file)
            for c in classes:
                print("image"+str(x)+" "+f'{labels.get(c.id, c.id)} {c.score:.5f}')
                if c.id != 0:
                    os.remove(str(data_dir)+"/image"+str(x)+'.jpg') #DELETING PHOTOS WITHOUT CLOUDS

        
    x += 1
    now_time = datetime.now() #UPDATING TIME 
camera.stop_preview() 