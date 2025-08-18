import sys
from typing import Optional

from groq import Groq
from pinecone import Pinecone

from settings import GROQ_API_KEY, PINECONE_API_KEY
from vector_db import search_similar

# --- Constantes de configuración ---
INDEX_NAME = "cv-index"
NAMESPACE = "cv-namespace"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- Inicialización del cliente Groq ---
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error al inicializar el cliente Groq: {e}", file=sys.stderr)
    sys.exit(1)


# --- Clase para la sesión de chat ---
class ChatSession:
    """
    Gestiona una sesión de chat con el modelo de lenguaje de Groq,
    utilizando un índice de Pinecone para agregar contexto.
    """
    def __init__(self, model: str = MODEL_NAME):
        """
        Inicializa la sesión de chat con los clientes de Groq y Pinecone.
        """
        self.groq_client = groq_client
        self.model = model
        self.messages = []
        try:
            self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            self.dense_index = self.pinecone_client.Index(INDEX_NAME)
        except Exception as e:
            print(f"Error al conectar con Pinecone: {e}", file=sys.stderr)
            sys.exit(1)

    def add_message(self, role: str, content: str):
        """Se agrega el mensaje a la historia de la conversación."""
        self.messages.append({"role": role, "content": content})

    def chat(
        self,
        user_message: str,
        temperature: float = 1,
        max_completion_tokens: int = 1024,
        top_p: float = 1,
        stream: bool = True,
        stop: Optional[str] = None
    ) -> str:
        """
        Se envía un mensaje al modelo de Groq con contexto de Pinecone y gestiona la respuesta.

        Args:
            user_message (str): El mensaje del usuario.
            ... (otros parámetros del API de Groq)

        Returns:
            str: La respuesta completa del asistente.
        """
        # 1. Obtener contexto relevante de Pinecone
        context_docs = search_similar(self.dense_index, user_message, top_k=3, debug=False)
        context_str = ' '.join(context_docs)

        # 2. Añadir el mensaje del usuario con el contexto
        full_user_message = f"{user_message}\n\nContexto: {context_str}"
        self.add_message("user", full_user_message)

        # 3. Llamar a la API de Groq
        completion = self.groq_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stream=stream,
            stop=stop
        )

        # 4. Procesar la respuesta y añadirla a la historia
        response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="")
            response += delta
        self.add_message("assistant", response)

        print()  # Nueva línea para mejor formato
        return response


# --- Script principal ---
if __name__ == "__main__":
    session = ChatSession()
    print("Iniciando sesión de chat. Escribe 'exit' o 'quit' para salir.")
    try:
        while True:
            user_input = input("Tú: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            session.chat(user_input)
    except KeyboardInterrupt:
        print("\n\nNos vemos la proxima.")
    except Exception as e:
        print(f"\nSe ha producido un error inesperado: {e}")
    finally:
        sys.exit(0)
        