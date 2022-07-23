import pickle 
import pandas as pd

app=Flask(__name__)


model= pickle.load(open("model.pk1","rb"))
scale=pickle.load(open("scale.pk1","rb"))


@app.route("/")
def hello_world():
    return render_template('index.html')
