# Chat con Groq y Pinecone

Este repositorio contiene un ejemplo de integración de un **modelo de lenguaje LLM (Groq)** con un índice de **Pinecone** para ofrecer respuestas enriquecidas con contexto (RAG).

---

## Descripción

El código implementa un **agente de chat** que:

1. Recibe mensajes del usuario.
2. Busca información relevante en un índice de Pinecone asociado a un CV.
3. Envía el mensaje junto con el contexto obtenido al modelo de Groq.
4. Devuelve la respuesta generada al usuario.

De esta forma, el LLM puede responder de manera más precisa basándose en datos específicos almacenados en Pinecone.

---

## Requisitos

* Python 3.10+

* Paquetes necesarios:

  ```bash
  pip install groq pinecone-client
  ```

* Credenciales API de:

  * Groq (`GROQ_API_KEY`)
  * Pinecone (`PINECONE_API_KEY`)

* Archivo de CVs ya cargado en el índice de Pinecone (`cv-index`) y la función `search_similar` definida para recuperar contexto.

---

## Configuración

```python
INDEX_NAME = "cv-index"
NAMESPACE = "cv-namespace"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
```

* `INDEX_NAME`: nombre del índice Pinecone donde se almacenan los CVs.
* `NAMESPACE`: namespace usado dentro del índice.
* `MODEL_NAME`: modelo de Groq que se usará para generar respuestas.

---

## Uso

1. Cargar el índice con vector_db.py e inicializar la sesión de chat:

```bash
python chat_groq.py
```

2. Escribir mensajes en la terminal:

```
Tú: ¿Cuál es la experiencia laboral de Martín?
```

3. El sistema buscará contexto en Pinecone y generará la respuesta usando Groq.

4. Para salir, escribir `exit` o `quit`.

---

## Funcionamiento Interno

* **ChatSession**: clase principal que gestiona la conversación y el historial.
* **add\_message()**: agrega un mensaje al historial (`user` o `assistant`).
* **chat()**: obtiene contexto de Pinecone y envía la conversación al LLM, mostrando la respuesta en tiempo real.
* **search\_similar()**: función que consulta el índice Pinecone y devuelve los chunks más relevantes.

---

## Logging y depuración

* Se imprimen en consola los resultados obtenidos de Pinecone y los fragmentos de contexto.
* Se pueden activar mensajes de depuración ajustando el parámetro `debug=True` en `search_similar`.


