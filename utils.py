"""
Funciones para crear e interacturar con el Modelo LLM mediante RAG.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os


def leer_pdf( path ):
    """Retorna el contenido del documento PDF.

    Params
    path: Ruta de la carpeta.

    Return
    pages: Lista de páginas dentro del documento.
    """
    try:
        loader = PyPDFLoader( path )
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        print(f'No se ha podido leer el archivo por el siguiente error: {e}')
        return None

def carpeta_vacia( path ):
    """Determina si una carpeta está vacía.

    Params
    path: Ruta de la carpeta.

    Return
    True o False, según corresponda.
    """
    if not os.path.exists(path):
        return True
    
    contenido = os.listdir(path)

    return len(contenido) == 0

def split_chunks( pdf ):
    """Divide en chunks el documento.

    Params
    pdf: Documento sobre el cuál se ejecuta el split.

    Return
    documents: Documento dividido en chunks.
    """
    # Cada párrafo tiene alrededor de 450 caracteres
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        length_function=len
        )
    documents = text_splitter.split_documents(pdf)

    return documents

def crear_vector_db( documents, embeddings, persist_directory ):
    """Crea una base de datos vectorial en ChromaDB.

    Params
    documents: Documentos a guardar.
    embeddings: Modelo de embeddings a utilizar.
    persist_directory: Directorio donde se guarda la base de datos.

    Return
    vectorstore: Base de datos vectorial.
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Una vez guardado los datos en Chroma, la data debería persistir en la ubicación indicada.
    # Sin embargo, es preferible asegurarse forzando esta acción
    vectorstore.persist()

    return vectorstore

def leer_vector_db( embeddings, persist_directory ):
    """Lee la base de datos vectorial previamente creada e ingestada.

    Params
    embeddings: Modelo de embeddings a utilizar.
    persist_directory: Directorio donde se encuentra la base de datos.

    Return
    vectorstore: Base de datos vectorial.
    """
    vectorstore = Chroma(persist_directory=persist_directory, 
                         embedding_function=embeddings)
    
    return vectorstore

def instanciar_modelo( openai_api_key ):
    """Crea un modelo de chat de OpenAI.

    Params
    openai_api_key: Secret API key para instanciar el modelo.

    Return
    chat: Modelo de OpenAI.
    """
    # Temperatura en 0.0 porque no queremos creatividad
    chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0 
    )

    return chat