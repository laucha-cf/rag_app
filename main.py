from flask import Flask, render_template, request, jsonify
from langchain_openai import OpenAIEmbeddings
from utils import carpeta_vacia, leer_pdf, split_chunks, crear_vector_db, leer_vector_db, instanciar_modelo
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

directorio = 'data'
path_pdf = 'data/data.pdf'
persist_directory = 'pi_db'
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

prompt_template = """
Contesta la pregunta basándote únicamente en este contexto:
{contexto}
---
La pregunta es la siguiente: {pregunta}.
La respuesta debe respetar estos puntos:
- Debe ser en el mismo idioma de la pregunta.
- Debe ser en una sola oración.
- Debe contener emojis al final que resuman el contenido de la oración.
- Debe ser en tercera persona.
"""

@app.route('/answer', methods=['POST'])
def get_answer():
    user_name = request.form.get('user_name')
    question = request.form.get('question')
    query_text = question

    # Si la carpeta está vacía, creo la Vector DB. Caso contrario, la leo
    if carpeta_vacia(path=directorio):
        pdf_document = leer_pdf(path=path_pdf)
        document_splited = split_chunks(pdf=pdf_document)
        vector_db = crear_vector_db(documents=document_splited, 
                                    embeddings=embeddings, 
                                    persist_directory=persist_directory)
    else:
        vector_db = leer_vector_db(embeddings=embeddings, 
                                   persist_directory=persist_directory)

    # En este punto ya tenemos la db

    results = vector_db.similarity_search_with_relevance_scores(query_text, k=3)
    # [ (Document(page_content='Contenido', 
    #             metadata={'source': 'path del contenido'}),
    #   score ), ... ]

    if len(results) == 0 or results[0][1] < 0.6:
        respuesta_texto = f'No se encontraron coincidencias con respecto a la pregunta: {query_text}'
    else:
        chat = instanciar_modelo(openai_api_key=openai_api_key)
        contexto = "\n\n---\n\n".join([doc.page_content.replace('\n', ' ') for doc, _score in results])
        prompt = prompt_template.format(contexto=contexto, pregunta=query_text)
        respuesta_texto = chat.invoke(prompt).content

    return jsonify({"user_name": user_name, "question": question, "answer": respuesta_texto})

@app.route('/')
def index():
    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)
