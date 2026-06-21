"""
Pipeline RAG com histórico de conversa.
Fluxo: condensa pergunta + histórico → retrieval → resposta com contexto.
"""
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ── Prompts ──────────────────────────────────────────────────────────────

# 1) Condensa a pergunta atual + histórico em uma pergunta autônoma.
#    Isso resolve referências como "esse assunto", "ele", "continue"…
CONDENSE_TEMPLATE = """Dado o histórico da conversa abaixo e a nova pergunta do usuário,
reformule a pergunta como uma pergunta completamente autônoma (sem pronomes ou referências
implícitas ao histórico). Se a pergunta já for autônoma, retorne-a sem alteração.

Histórico:
{chat_history}

Nova pergunta: {question}

Pergunta autônoma:"""

# 2) Responde usando os trechos recuperados + histórico completo (para coerência).
ANSWER_TEMPLATE = """Você é um assistente especializado em comportamento do consumidor.
Use APENAS os trechos de artigos acadêmicos fornecidos abaixo para responder à pergunta.
Se a resposta não estiver nos trechos, diga educadamente que não encontrou informação suficiente.

Sempre que possível, cite o artigo de origem (autor e ano) ao dar a resposta.
Responda em português, de forma clara e didática.

{chat_history}Trechos dos artigos:
{context}

Pergunta: {question}
Resposta:"""


# ── Funções públicas ──────────────────────────────────────────────────────

def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")


def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1000
    )


def load_vectorstore(vectorstore_path: str, embeddings: OpenAIEmbeddings) -> FAISS:
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def create_qa_chain(vectorstore: FAISS, llm: ChatOpenAI) -> dict:
    """Retorna um dict com o retriever e o LLM prontos para uso conversacional."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    return {"retriever": retriever, "llm": llm}


def _format_history(messages: list) -> str:
    """Formata o histórico de mensagens para inserir no prompt."""
    if not messages:
        return ""
    lines = ["Histórico da conversa:"]
    for msg in messages:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        # Trunca respostas longas para não explodir o contexto
        content = msg["content"]
        if len(content) > 600:
            content = content[:600] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n\n"


def answer_question(
    qa_chain: dict,
    question: str,
    chat_history: list | None = None
) -> dict:
    """
    Responde a uma pergunta com contexto conversacional.

    chat_history: lista de dicts {"role": "user"|"assistant", "content": str}
                  com as mensagens ANTERIORES à pergunta atual.
    """
    retriever = qa_chain["retriever"]
    llm = qa_chain["llm"]
    history = chat_history or []

    # 1) Condensar a pergunta se houver histórico
    if history:
        history_str = _format_history(history)
        condense_prompt = CONDENSE_TEMPLATE.format(
            chat_history=history_str,
            question=question
        )
        standalone = llm.invoke(condense_prompt).content.strip()
    else:
        standalone = question
        history_str = ""

    # 2) Recuperar chunks relevantes com a pergunta autônoma
    docs = retriever.invoke(standalone)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3) Responder com contexto + histórico
    history_block = _format_history(history)
    answer_prompt = ANSWER_TEMPLATE.format(
        chat_history=history_block,
        context=context,
        question=question      # usa a pergunta original (mais natural)
    )
    answer = llm.invoke(answer_prompt).content.strip()

    # Extrai fontes únicas
    sources = sorted({
        doc.metadata.get("source", "Desconhecido") for doc in docs
    })

    return {"answer": answer, "sources": sources}
