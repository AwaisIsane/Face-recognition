from flask import Flask,render_template,request
import numpy as np
import cv2
from utils1 import add_to_database,check_in_database,load_face_model,names_in_database
database = np.load('database.npy',allow_pickle=True)
database = database[()]
model = load_face_model()
print("loaded face model")
app = Flask(__name__)
@app.route('/')
def form():
    return render_template("index.html") 


@app.route('/names')
def names():
    names = names_in_database(database)
    return render_template("names.html", prediction = names)
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        username = request.form["name"]
        file = request.files['filename1'].read() ## byte file
        npimg = np.fromstring(file, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        while True :
            if request.form['submit_button'] == 'add_to_database':
                check = add_to_database(img,username,database,model)
                np.save("database.npy",database)                
                return render_template("result.html", prediction = check)
            elif request.form['submit_button'] == 'check_in_database':
                try:
                    prediction=check_in_database(img,username,database,model)
                    return render_template("result.html", prediction = prediction)
                except KeyError:
                    return render_template("result.html", prediction = "username not found")
            elif request.form['submit_button'] == 'delete_from_database':
               try: 
                    del database[username]
                    np.save("database.npy",database)  
                    return render_template("result.html", prediction = "DEleted")
               except KeyError:
                    return render_template("result.html", prediction = "Key already deleted")
                    
    return render_template("result.html", prediction = "exited") 

# When debug = True, code is reloaded on the fly while saved3
if __name__ == '__main__':
    app.run()