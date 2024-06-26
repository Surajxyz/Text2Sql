#app.py
from flask import Flask,request, render_template
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__,template_folder='Template')

app.secret_key = 'your_secret_key'

@app.route("/",methods=["GET","POST"])
def login():
    return render_template("index.html")

@app.route('/upload', methods=["GET",'POST'])
def upload_file():
    Schema=str(request.form.get("input1"))
    Text=str(request.form.get("input2"))
    model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    try:
        if request.form.get("btnTask1")=="click1":
            input_text = " ".join(["Question: ",Text, "Schema:", Schema])
            model_inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**model_inputs, max_length=512)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return render_template("index.html",answer=f"Sql Query: {output_text}",Text=f"Query in NLP: {Text}",Schema=f"Table Schema:  {Schema}")
        
    except Exception as e:
        return render_template("index.html",answer=e)
    

if __name__ == '__main__':
    app.run(debug=True)
        


