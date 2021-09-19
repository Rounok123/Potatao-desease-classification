from fastapi import FastAPI, File, UploadFile
from tensorflow.python.ops.gen_lookup_ops import initialize_table_from_text_file_v2_eager_fallback
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:

    image = np.array(Image.open(BytesIO(data)))  # Take will take uploaded image as byte and converting it to an image and going converting it to an numpy array for prediction
    return image


@ app.post("/predict")
async def predict(file: UploadFile = File(...)): # It means the func will take input file only
    
    image = read_file_as_image(await file.read()) # Reading the data from the uploaded file using the function
    img_batch = np.expand_dims(image, 0) # As our model image batch rather than a sigle image we must make it 2D from 1D, So expand_dims help to adding more dimensions
    
    predictions = MODEL.predict(img_batch)  # Doing Prediction using the saved model
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])] # taking the highest value of the first prediction and using that as a index of the ClASS_NAME array to predict the class_name 

    confidence = np.max(predictions[0]) # Taking the highest value of prediction as cofidence

    return{
        'class': predicted_class,     # Returning the predicted desease and confidence/accuracy
        'confidence': float(confidence) 
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
