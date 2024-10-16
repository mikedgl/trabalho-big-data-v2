from flask import Flask, request, jsonify
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Responda à pergunta com base apenas no seguinte contexto:

{context}

---

Responda à pergunta com base no contexto acima: {question}
"""

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_rag():
    # Obtenha a pergunta do JSON
    data = request.json
    query_text = data.get('query_text')

    # Prepare o DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Pesquisa no DB
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3.1")
    response_text = model.invoke(prompt)

    return jsonify({"resposta": response_text})

if __name__ == "__main__":
    app.run(debug=True)