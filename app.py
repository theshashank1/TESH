from flask import Flask, request, jsonify, render_template
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = Ollama(model="gemma:2b")

# General purpose prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])


@app.route('/')
def index() :
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET' :
        return render_template('login.html')


@app.route('/api/process', methods=['POST'])
def process() :
    data = request.json
    user_input = data['text']

    # Create the formatted prompt
    formatted_prompt = prompt_template.format(input=user_input)

    # Get the response from the LLM
    response = llm(formatted_prompt)

    # Parse the response using StrOutputParser
    output_parser = StrOutputParser()
    parsed_output = output_parser.parse(response)

    return jsonify({'response' : parsed_output})


if __name__ == '__main__' :
    app.run(debug=True)
