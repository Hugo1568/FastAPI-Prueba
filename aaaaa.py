from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import numpy as np
from fastapi.staticfiles import StaticFiles
import shutil



# Inicializamos fastApi
app = FastAPI()
# Para recibir un archivo en este caso una imagen
@app.post("/file")
async def _file_upload(
    my_file: UploadFile = File(...),
):
    # Tomamos el archivo para eso es el .file
    img_path = my_file.file


    # Esto se hizo para saber si realmente recibiamos la imagen
    # Esto abre una imagen con pillow
    #im = Image.open(my_file.file)
    #im.show()
    
    with open("destination.png", "wb") as buffer:
        shutil.copyfileobj(my_file.file, buffer)

 
    return {
        'Nombre': my_file.filename
    }