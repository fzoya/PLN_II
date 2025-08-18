import sys
import time
import logging
from pinecone import Pinecone
from settings import PINECONE_API_KEY
from chunk import read_and_chunk_sentences

# -------------------------------
# Configuración global
# -------------------------------
INDEX_NAME = "cv-index"
NAMESPACE = "cv-namespace"
CHUNK_SIZE = 3
CHUNK_OVERLAP = 1
LOAD_DATA = True

# Logging en vez de prints
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Inicialización cliente
pc = Pinecone(api_key=PINECONE_API_KEY)
logging.info(f"API KEY detectada: {PINECONE_API_KEY[:6]}... (ocultada por seguridad)")


# -------------------------------
# Crear índice si no existe
# -------------------------------
def ensure_index(name: str):
    """Crea el índice si no existe."""
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if name not in existing_indexes:
        logging.info(f"Índice {name} no encontrado. Creando...")
        pc.create_index_for_model(
            name=name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
        # Le damos tiempo a Pinecone a levantarlo
        time.sleep(5)
        logging.info(f"Índice {name} creado con éxito.")
    else:
        logging.info(f"Índice {name} ya existe.")

    return pc.Index(name)


def load_data(index):
    """Lee y sube los chunks al índice de a uno."""
    logging.info("Leyendo y dividiendo CV...")
    
    # Se mantienen las constantes para la división
    chunks = read_and_chunk_sentences("cv1.txt", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    logging.info(f"{len(chunks)} chunks generados.")
    
    # 1. Se crean los registros uno a uno
    for i, chunk in enumerate(chunks):
        record = {
            "_id": f"cv_chunk_{i+1}",
            "chunk_text": chunk,
            "category": "cv"
        }
        
        # 2. Se inserta cada registro individualmente
        index.upsert_records(namespace=NAMESPACE, records=[record])
        logging.info(f"Registro cv_chunk_{i+1} subido.")

    logging.info("Carga de todos los registros completada.")
    
    # 3. Se espera un breve momento para la indexación
    time.sleep(5)


# -------------------------------
# Búsqueda
# -------------------------------
def search_similar(index, text, top_k=3, namespace=NAMESPACE, debug=True):
    """Busca en el índice los registros más similares al texto dado."""
    stats = index.describe_index_stats()
    logging.info(f"Estadísticas del índice: {stats}")

    results = index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "inputs": {"text": text}
        }
    )

    data = []
    for hit in results["result"]["hits"]:
        tmp = (
            f"id: {hit['_id']:<10} | "
            f"score: {round(hit['_score'], 2):<5} | "
            f"category: {hit['fields']['category']:<10} | "
            f"text: {hit['fields']['chunk_text'][:80]}..."
        )
        if debug:
            print(tmp)
        data.append(tmp)
    return data


# -------------------------------
# Main loop
# -------------------------------
if __name__ == "__main__":
    try:
        dense_index = ensure_index(INDEX_NAME)

        if LOAD_DATA:
            load_data(dense_index)

        while True:
            msg = input("Texto a buscar (Enter para salir): ").strip()
            if not msg:
                logging.info("Saliendo...")
                break
            search_similar(dense_index, msg)

    except KeyboardInterrupt:
        logging.info("Interrumpido por el usuario. Saliendo...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error inesperado: {e}", exc_info=True)
        sys.exit(1)
