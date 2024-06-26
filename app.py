#app.py
from flask import Flask, request, render_template
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__,template_folder='Template')

app.secret_key = 'your_secret_key'

def generate_sql(input_prompt):
    # Tokenize the input prompt
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql')
    model = model.to(device)
    model.eval()
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    # Decode the output IDs to a string (SQL query in this case)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql

@app.route("/",methods=["GET","POST"])
def login():
    return render_template("index.html")

@app.route('/upload', methods=["GET",'POST'])
def upload():
    Schema=str(request.form.get("input1"))
    Text=str(request.form.get("input2"))
    try:
        if request.form.get("btnTask1")=="click1":
            
            input_prompt = "tables:\n" + Schema+ "\n"  "query for:" +  Text
            generated_sql = generate_sql(input_prompt)

            return render_template("index.html",answer=f"Sql Query: {generated_sql}",Text=f"Query in NLP: {Text}",Schema=f"Table Schema:  {Schema}")
    
    except Exception as e:
        return render_template("index.html",answer=e)
    

if __name__ == '__main__':
    app.run(debug=True)
        


