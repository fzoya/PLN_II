"""
cv_agent.py
Agente para seleccionar el índice (CV) correcto en Pinecone en base a chunks relevantes
y responder usando un LLM (Groq).

Requisitos:
  - pip install groq pinecone
  - Archivo settings.py con GROQ_API_KEY y PINECONE_API_KEY

Ejecución:
  python cv_agent.py
"""
import sys
import time
import logging
from typing import List, Dict

from groq import Groq
from pinecone import Pinecone

from settings import GROQ_API_KEY, PINECONE_API_KEY

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
NAMESPACE = "cv-namespace"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Inicialización de clientes
# -----------------------------------------------------------------------------
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error al inicializar Groq: {e}", file=sys.stderr)
    sys.exit(1)

try:
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"Error al conectar con Pinecone: {e}", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def search_similar(index, text: str, top_k: int = 3, namespace: str = NAMESPACE, debug: bool = True) -> List[Dict]:
    """
    Busca en el índice los registros más similares al texto dado y retorna una lista
    de dicts estructurados con id, score, category y text.

    NOTA: Esta implementación asume la respuesta estilo:
        {
          "result": {
            "hits": [
              {"_id": "...", "_score": ..., "fields": {"category": "...", "chunk_text": "..."}},
              ...
            ]
          }
        }
    Si tu respuesta difiere, ajusta los accesos a las claves aquí.
    """
    try:
        results = index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": text}
            }
        )
    except Exception as e:
        logging.error(f"Error en index.search: {e}")
        return []

    data: List[Dict] = []
    try:
        for hit in results["result"]["hits"]:
            entry = {
                "id": hit.get("_id", ""),
                "score": round(float(hit.get("_score", 0.0)), 2),
                "category": hit.get("fields", {}).get("category", ""),
                "text": hit.get("fields", {}).get("chunk_text", ""),
            }
            if debug:
                logging.info(entry)
            data.append(entry)
    except Exception as e:
        logging.error(f"Error parseando resultados de Pinecone: {e}")
    return data


# -----------------------------------------------------------------------------
# Definición de Tools
# -----------------------------------------------------------------------------
class Tool:
    """Interfaz base para todas las Tools."""
    def run(self, *args, **kwargs):
        raise NotImplementedError


class CVSelectorTool(Tool):
    """
    Tool que selecciona el índice (CV) más relevante según los chunks asociados,
    usando los embeddings ya almacenados en Pinecone y un LLM para decidir.
    """
    def __init__(self, pinecone_client: Pinecone, llm_client: Groq, model_name: str):
        self.pinecone = pinecone_client
        self.llm = llm_client
        self.model_name = model_name

    def run(self, query: str) -> str:
        # Listar índices disponibles en Pinecone
        try:
            indexes = [idx.name for idx in self.pinecone.list_indexes()]
        except Exception as e:
            logging.error(f"No fue posible listar índices de Pinecone: {e}")
            return ""

        if not indexes:
            logging.warning("No hay índices disponibles en Pinecone.")
            return ""

        cv_chunks: Dict[str, List[str]] = {}

        # Recuperar los chunks más relevantes de cada índice
        for idx in indexes:
            logging.info(f"🔍 Buscando en índice: {idx}")
            try:
                index = self.pinecone.Index(idx)
            except Exception as e:
                logging.error(f"No se pudo abrir el índice '{idx}': {e}")
                continue

            # Evitar rate limits simples
            time.sleep(0.5)

            results = search_similar(index, query, top_k=3, debug=False)

            # Guardar solo el texto de los chunks (para reducir tokens)
            chunk_texts = [c.get("text", "") for c in results if c.get("text")]
            if not chunk_texts:
                # Si no hubo hits, igual registrar vacío para que el LLM lo considere
                chunk_texts = ["(sin coincidencias relevantes)"]
            cv_chunks[idx] = chunk_texts

        # Construir prompt para el LLM
        system_prompt = "Tienes estos índices con sus contenidos más relevantes:\n"
        for idx, chunks in cv_chunks.items():
            # Limitar cada índice a ~3 snippets cortos
            snippets = " | ".join(chunks[:3])
            system_prompt += f"- Índice: {idx}\n  Contenido: {snippets}\n"
        system_prompt += (
            "\nInstrucción: Dada la consulta del usuario, responde SOLO con el NOMBRE EXACTO "
            "del índice más relevante. No incluyas texto adicional."
        )

        # Llamar al LLM
        try:
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                # Para Groq, usa max_tokens (no max_completion_tokens)
                max_tokens=50,
                temperature=0
            )
            # Extraer respuesta
            # Extraer el contenido del primer choice
            if hasattr(completion.choices[0], "message"):
                selected_idx = completion.choices[0].message.content.strip()
            elif hasattr(completion.choices[0], "delta"):
                selected_idx = completion.choices[0].delta.content.strip()
            else:
                selected_idx = str(completion.choices[0]).strip()  # fallback genérico

            print(f"Indice Seleccionado: {selected_idx}")
            #selected_idx = completion.choices[0].message["content"].strip()
            #selected_idx = completion.choices[0].content.strip()

            
        except Exception as e:
            logging.error(f"Error llamando al LLM para seleccionar índice: {e}")
            return ""

        logging.info(f"✅ Índice seleccionado por el LLM: {selected_idx}")
        return selected_idx if selected_idx in cv_chunks else ""


class CVRetrieverTool(Tool):
    """
    Tool que recupera el contexto desde el índice específico del CV.
    """
    def __init__(self, pinecone_client: Pinecone):
        self.pinecone = pinecone_client

    def run(self, index_name: str, query: str, top_k: int = 5) -> List[Dict]:
        try:
            index = self.pinecone.Index(index_name)
        except Exception as e:
            logging.error(f"No se pudo abrir el índice '{index_name}': {e}")
            return []
        return search_similar(index, query, top_k=top_k, debug=False)


class LLMResponderTool(Tool):
    """
    Tool que genera la respuesta final usando Groq.
    """
    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model = model

    def run(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        except Exception as e:
            logging.error(f"Error llamando al LLM para responder: {e}")
            return "Ocurrió un error generando la respuesta."

        response = ""
        for chunk in completion:
            # Manejar la llegada de deltas de forma segura
            delta = ""
            try:
                delta = getattr(chunk.choices[0].delta, "content", "") or ""
            except Exception:
                pass
            if delta:
                print(delta, end="", flush=True)
                response += delta
        print()  # salto de línea al final del stream
        return response


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class CVChatAgent:
    """
    Agente que identifica el CV correcto (índice) y responde usando su contexto.
    """
    def __init__(self, model: str = MODEL_NAME):
        self.messages: List[Dict] = []
        self.groq_client = groq_client
        self.pinecone_client = pinecone_client

        # Tools
        self.cv_selector = CVSelectorTool(self.pinecone_client, self.groq_client, MODEL_NAME)
        self.cv_retriever = CVRetrieverTool(self.pinecone_client)
        self.llm_responder = LLMResponderTool(self.groq_client, model)

    def chat(self, user_message: str) -> str:
        # Paso 1: Identificar índice correspondiente al CV que más probabilidad tiene de contener la respuesta
        target_index = self.cv_selector.run(user_message)
        if not target_index:
            logging.warning("No se pudo identificar el CV en la base de datos.")
            return "No encontré un CV relevante para tu consulta."

        # Paso 2: Recuperar contexto desde el índice seleccionado
        context_docs = self.cv_retriever.run(target_index, user_message, top_k=3)

        # Preparar un resumen corto de contexto para no gastar demasiados tokens
        context_snippets = []
        for doc in context_docs[:5]:
            txt = doc.get("text", "")
            if txt:
                # recortar por si vienen textos largos
                context_snippets.append(txt[:300])
        context_blob = "\n- ".join(context_snippets) if context_snippets else "(sin contexto relevante)"

        # Paso 3: Crear mensajes con contexto
        system_msg = (
            "Eres un asistente que responde preguntas basándote EXCLUSIVAMENTE en el contexto provisto. "
            "Si el contexto no contiene la respuesta, reconoce la limitación y sugiere preguntar de otra forma."
        )
        user_msg = (
            f"Consulta del usuario: {user_message}\n\n"
            f"Índice seleccionado: {target_index}\n"
            f"Contexto (fragmentos):\n- {context_blob}\n\n"
            "Responde de forma concisa y útil."
        )

        self.messages.append({"role": "system", "content": system_msg})
        self.messages.append({"role": "user", "content": user_msg})

        # Paso 4: Generar respuesta con Groq
        response = self.llm_responder.run(self.messages)
        self.messages.append({"role": "assistant", "content": response})

        return response


# -----------------------------------------------------------------------------
# Script principal
# -----------------------------------------------------------------------------
def main():
    agent = CVChatAgent()
    print("Agente de consulta de CVs iniciado. Escribe 'exit' para salir.\n")

    try:
        while True:
            user_input = input("Tú: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue
            print("\n--- Respuesta ---")
            _ = agent.chat(user_input)
            print("\n-----------------\n")
    except KeyboardInterrupt:
        print("\n\nCerrando agente...")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
