import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boutine']

app = Flask(__name__)
model = tf.keras.models.load_model('./static/model/Model.h5')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # return jsonify({'prediction': "hello its post"})
        #print("hello")
        data = request.files['uploaded-file']
        data.save("./static/uploads/" + data.filename)
        imagePath = "./static/uploads/" + data.filename
        img = Image.open(imagePath)
        img = img.resize((28, 28), Image.ANTIALIAS).convert(mode='L')
        img_arr = np.array(img).reshape(1, 28, 28, 1)
        img_arr = img_arr.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(img_arr)

        index_label = np.argmax(pred[0], axis=-1)
        print(pred)
        return render_template("index.html", data={"image": data.filename, "label": class_names[index_label]})

    return render_template("index.html")


app.run()