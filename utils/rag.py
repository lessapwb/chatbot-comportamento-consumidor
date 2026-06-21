"""
Pipeline RAG com histórico de conversa e multi-query retrieval.
Fluxo: condensa pergunta + histórico → multi-query retrieval → resposta com contexto.
"""
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ── Prompts ──────────────────────────────────────────────────────────────

CONDENSE_TEMPLATE = """Dado o histórico da conversa abaixo e a nova pergunta do usuário,
reformule a pergunta como uma pergunta completamente autônoma (sem pronomes ou referências
implícitas ao histórico). Se a pergunta já for autônoma, retorne-a sem alteração.

Histórico:
{chat_history}

Nova pergunta: {question}

Pergunta autônoma:"""

# Gera variações da pergunta para cobrir mais ângulos nos artigos
MULTI_QUERY_TEMPLATE = """Você é um assistente de pesquisa acadêmica em comportamento do consumidor.
Dada a pergunta abaixo, escreva 2 reformulações alternativas que busquem aspectos
diferentes ou complementares do mesmo tema em uma base de artigos científicos.
Retorne apenas as 2 reformulações, uma por linha, sem numeração ou prefixos.

Pergunta: {question}

Reformulações:"""

ANSWER_TEMPLATE = """Você é um assistente acadêmico especializado em comportamento do consumidor.

Você tem acesso a trechos de artigos acadêmicos (listados abaixo) e também ao seu conhecimento geral sobre a área.

Diretrizes:
- Priorize sempre as informações dos artigos fornecidos, citando autor e ano quando relevante.
- Quando os trechos não cobrirem completamente a pergunta, use seu conhecimento acadêmico para complementar — mas sinalize claramente quando estiver indo além dos artigos ("Com base no conhecimento da área..." ou "Além dos artigos indexados...").
- Para pedidos criativos ou propositivos (sugerir temas de pesquisa, propor modelos, recomendar abordagens), elabore ativamente usando os artigos como base teórica e fundamente suas sugestões.
- Seja específico, didático e útil. Nunca recuse ajudar por falta de informação nos trechos se você puder raciocinar com o que tem.
- Responda sempre em português.

{chat_history}Trechos dos artigos indexados:
{context}

Pergunta: {question}
Resposta:"""


# ── Funções públicas ──────────────────────────────────────────────────────

def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")


def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=4000
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
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
    )
    return {"retriever": retriever, "llm": llm}


def _format_history(messages: list) -> str:
    """Formata o histórico de mensagens para inserir no prompt."""
    if not messages:
        return ""
    lines = ["Histórico da conversa:"]
    for msg in messages:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        content = msg["content"]
        if len(content) > 2000:
            content = content[:2000] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n\n"


def _multi_query_docs(retriever, llm, question: str) -> list:
    """
    Gera 2 reformulações da pergunta via LLM e une os chunks
    recuperados pelas 3 buscas (original + 2 variações), deduplicados.
    Fallback para busca simples se a geração falhar.
    """
    try:
        raw = llm.invoke(MULTI_QUERY_TEMPLATE.format(question=question)).content.strip()
        variants = [q.strip() for q in raw.split("\n") if q.strip()][:2]
    except Exception:
        variants = []

    seen: set[str] = set()
    docs: list = []
    for q in [question] + variants:
        try:
            for doc in retriever.invoke(q):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
        except Exception:
            continue
    return docs


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

    # 2) Multi-query retrieval: 3 buscas MMR → chunks deduplicados
    docs = _multi_query_docs(retriever, llm, standalone)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3) Responder com contexto + histórico
    history_block = _format_history(history)
    answer_prompt = ANSWER_TEMPLATE.format(
        chat_history=history_block,
        context=context,
        question=question
    )
    answer = llm.invoke(answer_prompt).content.strip()

    sources = sorted({
        doc.metadata.get("source", "Desconhecido") for doc in docs
    })

    return {"answer": answer, "sources": sources}
