from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
# dump information to that file
clf = pickle.load(file)

# close the file
file.close()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if (request.method == "POST"):
        mydict=request.form
        lsepal=float(mydict['lsepal'])
        wsepal=float(mydict['wsepal'])
        lpetal=float(mydict['lpetal'])
        wpetal=float(mydict['wpetal'])
        # code for inference
        inputfeatures=[lsepal, wsepal, lpetal, wpetal]
        # print(inputfeatures)
        flower=clf.predict([inputfeatures])
        # print(flower)
        if(flower[0]==0):
            fw="Iris-setosa"
        elif(flower[0]==1):
            fw="Iris-versicolor"
        else:
            fw="Iris-virginica"


        # render on the template
        return render_template('show.html', fw=fw)
    return render_template('index.html')

    # return 'Hello, World!'+str(corona)



if __name__ == "__main__":
    app.run(debug=True)