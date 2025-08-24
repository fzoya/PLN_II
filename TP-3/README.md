# Búsqueda de info de CV's mediante Pinecone y Groq

Este repositorio contiene un script en Python para **gestionar y consultar CVs** mediante **Pinecone**, un vector database. El flujo principal incluye scripts para lectura de archivos de CV, segmentación en "chunks", almacenamiento en un índice vectorial y búsqueda semántica mediante cliente Groq.

---

## Funcionalidades

1. **Creación de índice en Pinecone** [vector_db.py]
   - El script verifica si el índice existe; si no, lo crea automáticamente.
   - Utiliza el modelo de embeddings `multilingual-e5-large` para generar vectores semánticos a partir del texto de los CVs.

2. **Carga de datos**
   - Lee un archivo de CV (Ejemplo `cv1.txt`) y lo divide en chunks de tamaño configurable (`CHUNK_SIZE`) con solapamiento (`CHUNK_OVERLAP`).
   - Cada chunk se inserta individualmente en el índice Pinecone con su ID y categoría.

3. **Búsqueda semántica**
   - Permite consultar el índice con cualquier texto.
   - Devuelve los chunks más similares según el embedding generado, mostrando ID, score, categoría y un fragmento del contenido.
   - Configurable el número de resultados a retornar (`top_k`).

4. **Logging y manejo de errores**
   - Uso de `logging` para informar sobre creación de índices, subida de registros y resultados de búsqueda.
   - Manejo de interrupciones del usuario y errores inesperados.

---

## Configuración

Antes de ejecutar el script:

1. Instalar las dependencias
   
2. Configura las variables de entorno con las API Key de Pinecone y Groq, obtenidas mediante `settings.py`
   
3. Dispone de los archivos de texto, denominados `cv1-3.txt`, en el mismo directorio, conteniendo el texto de los CVs a indexar.

4. Ajusta parámetros opcionales en el script de carga [vector_db.py]:
   - `INDEX_NAME`: nombre del índice Pinecone.
   - `NAMESPACE`: namespace usado para los registros.
   - `CHUNK_SIZE` y `CHUNK_OVERLAP`: tamaño y solapamiento de los chunks.

---

## Uso

Ejecuta el script de carga de los archivos para cada uno de ellos editando los nombres de index y file:
```bash
python vector_db.py
```

Ejecuta el script de consulta de CV's en el mismo directorio
```bash
python cv_agents.py
```

## Ejemplo de búsqueda

```
Texto a buscar (Enter para salir): Ingeniera en sistemas con experiencia en DevOps
id: cv_chunk_2  | score: 0.89  | category: cv         | text: Experiencia en automatización de pipelines CI/CD con Jenkins y GitLab...
id: cv_chunk_5  | score: 0.75  | category: cv         | text: Gestión de microservicios en Kubernetes y Docker...
id: cv_chunk_1  | score: 0.65  | category: cv         | text: Coordinación de equipos ágiles y mejora de procesos...
```

---

## Estructura del repositorio

```
.
├── cv_agents.py        # Script principal de indexación y búsqueda
├── settings.py         # Configuración de API Keys
├── cv1.txt             # Archivos de texto con CV (de 1 a 3)
├── cv2.txt             # Archivos de texto con CV (de 1 a 3)
├── cv3.txt             # Archivos de texto con CV (de 1 a 3)
├── vector_db.py        # Script para la creacion y carga de los indices con los chunks de los CVs
├── chunk.py            # Script con funcion de genercioon de chunks
└── README.md           # Documentación del proyecto
```

---

## Notas

- Se recomienda **no exceder los límites de la API de Pinecone** 
- El modelo de embeddings puede cambiarse según la necesidad, actualmente se usa `multilingual-e5-large` por falta de crédito disponible con modelo llama.

---

## Autor

Proyecto desarrollado por Federico M. Zoya.

