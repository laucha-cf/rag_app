"""
Ejecución principal.
Instanciamos la VectorDB, el Modelo LLM y realizamos consultas.
"""

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from utils import *
import os

# Credencial OpenAI
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

directorio = 'data'
path_pdf = 'data/data.pdf' #Ubicación del documento
persist_directory = 'pi_db' # Ubicación base de datos vectorial

# Declaramos el modelo de embeddings a utilizar
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Plantilla del Prompt
prompt_template = """
Contesta la pregunta basándote únicamente en este contexto:

{contexto}

---
La pregunta es la siguiente: {pregunta} .

La respuesta debe respetar estos puntos:
- No debe ser inventada. Solo contenido basado en el contexto que te brindé.
- Debe ser en una sola oración.
- Debe ser en el mismo idioma de la pregunta.
- Debe contener emojis que resuman el contenido de la oración.
- Debe ser en tercera persona.
"""

query_text = "What did Emma decided to do?"


if __name__ == "__main__":

    if carpeta_vacia( path=directorio ):
        pdf_document = leer_pdf( path=path_pdf )
        document_splited = split_chunks( pdf=pdf_document )
        vector_db = crear_vector_db( documents=document_splited,
                                     embeddings=embeddings,
                                     persist_directory=persist_directory )
    else:
        #La vector db ya existe
        vector_db = leer_vector_db( embeddings=embeddings,
                                    persist_directory=persist_directory )


    # En este punto ya tenemos la vector db para realizar las consultas a OpenAI
    

    # Realizamos la búsqueda de los 3 vectores más similares con sus respectivos scores
    results = vector_db.similarity_search_with_relevance_scores( query_text, k=3 )


    # Respondemos a la pregunta solo si la coincidencia del primer vector es 0.7 o mayor
    if len(results)==0 or results[0][1] <= 0.7:
        respuesta_texto = f'No se encontraron coincidencias con respecto a la pregunta: {query_text}'
    else:
        chat = instanciar_modelo( openai_api_key=openai_api_key )
        contexto = "\n\n---\n\n".join( [doc.page_content.replace('\n', ' ') for doc, _score in results] )
        prompt = prompt_template.format(contexto=contexto, pregunta=query_text)
        respuesta_texto = chat.invoke( prompt ).content


    print(f'La respuesta es : {respuesta_texto}')