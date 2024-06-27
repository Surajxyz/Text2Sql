# Text2Sql
The objective of this assignment is to develop a system capable of converting natural language queries into SQL (Structured Query Language) statements using Generative AI techniques. 

# Creation of Environment

```python -m .venv venv```

# Install requirement.txt

```pip install -r requirement.txt```

# Creation of index.html
css code and js code is available in the index.html file only 

# Creation of app.py and app2.py

# app.py

app.py contains use of the 'cssupport/t5-small-awesome-text-to-sql' model for the creation of sql query from the text which is fined tuned in file Fine_tuning_for_text2sql_model.ipynb

Schema for this 

```Create table <Table name1> (Column_name1 datatype, column_name2 datatype...); Create table <Table name 2>(Column_name1 datatype, column_name2 datatype...)```

# app2.py

app2.py contains use of the fine tuned model 'gaussalgo/T5-LM-Large-text2sql-spider' on spider datset for the creation odf sql query from the text given

Schema for app2.py look like this 

```table_name column1_name column1_type column2_name column2_type ... foreign_key: FK_name FK_type from table_name column_name primary key: column_name [SEP]  table_name2 ...```

# Fine_tuning_for_text2sql_model.ipynb file 
It is used to fine tuned 't5-small' using 

https://huggingface.co/datasets/b-mc2/sql-create-context
https://huggingface.co/datasets/Clinton/Text-to-sql-v1
https://huggingface.co/datasets/knowrohit07/know_sql

datasets and taining occured for the test set instead of validation set because of the compute problem and we get the training loss of 0.076000 and validation loss of 0.053022


