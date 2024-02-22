### Retrieval Augmented Generation App

![App Image](https://miro.medium.com/v2/resize:fit:1400/0*Ko_ihY8ecAukf2g1.png)

App que permite hacer consultas mediante una API para interactuar con un LLM con el fin de generar una respuesta (sobre un documento en particular) a la pregunta brindada por el usuario.

El modelo utilizado se instancia con el objetivo de brindar respuestas concretas, concisas y con la menor creatividad posible.

#### Instrucciones de Ejecución:

1. Crear un entorno virtual.
2. Instalar las librerías de Python listadas en `requirements.txt`.
3. Crear un archivo .env y una variable llamada OPENAI_API_KEY que contenga la secret access key de openai.
4. Ejecutar el siguiente comando dentro de la carpeta con el proyecto:

```bash
python main.py
