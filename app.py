"""
Chatbot de Comportamento do Consumidor
--------------------------------------
Assistente virtual que conversa sobre artigos acadêmicos de
comportamento do consumidor usando RAG (Retrieval-Augmented Generation).

Execute com: streamlit run app.py
"""
import os
import re
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rag import (
    get_embeddings,
    get_llm,
    load_vectorstore,
    create_qa_chain,
    answer_question
)

# ── Configuração da página — deve vir ANTES de qualquer st.* call ──────
st.set_page_config(
    page_title="Chatbot • Comportamento do Consumidor",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="auto"
)

# ── API Key (secrets > env > erro) ────────────────────────────────────
def get_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    st.error("Chave OpenAI não encontrada. Configure os Secrets no Streamlit Cloud.",
             icon=":material/key:")
    st.stop()

API_KEY = get_api_key()

# ── CSS personalizado (alto contraste, moderno) ────────────────────────
st.markdown("""
<style>
    /* Biblioteca de ícones gratuita: Material Symbols (Google).
       Usada APENAS nos meus <span> custom (.material-symbols-rounded).
       Os ícones :material/...: do Streamlit usam a fonte própria dele. */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap');

    span.material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal;
        font-style: normal;
        line-height: 1;
        vertical-align: middle;
        -webkit-font-smoothing: antialiased;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ================================================================
       FUNDO
       ================================================================ */
    .stApp {
        background: linear-gradient(to bottom, #f8fafc 0%, #eef2f7 100%);
    }

    /* ================================================================
       CABEÇALHO PRINCIPAL
       ================================================================ */
    .main-header {
        text-align: center;
        padding: 2.2rem 1rem 0.6rem 1rem;
        animation: fadeInUp 0.5s ease-out;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.3rem;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
    }
    .main-header h1 .material-symbols-rounded {
        font-size: 2.4rem;
        color: #2563eb;
        font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 40;
    }
    .main-header .subtitle {
        color: #475569;          /* contraste reforçado */
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* ================================================================
       CHAT — container centralizado
       ================================================================ */
    .chat-container {
        max-width: 820px;
        margin: 0 auto;
        padding: 1rem;
    }
    [data-testid="stChatMessage"] {
        margin-bottom: 1rem;
        animation: fadeInUp 0.35s ease-out;
        background: transparent !important;
    }

    /* Mensagem do USUÁRIO (bolha azul) — diferenciada pelo aria-label */
    [data-testid="stChatMessageContent"][aria-label*="from user"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 14px 20px !important;
        line-height: 1.6 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.22) !important;
    }
    [data-testid="stChatMessageContent"][aria-label*="from user"],
    [data-testid="stChatMessageContent"][aria-label*="from user"] * {
        color: #ffffff !important;
    }

    /* Mensagem do ASSISTENTE (cartão branco, texto escuro) */
    [data-testid="stChatMessageContent"][aria-label*="from assistant"] {
        background: #ffffff !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 14px 20px !important;
        line-height: 1.7 !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
    }
    [data-testid="stChatMessageContent"][aria-label*="from assistant"],
    [data-testid="stChatMessageContent"][aria-label*="from assistant"] p,
    [data-testid="stChatMessageContent"][aria-label*="from assistant"] li,
    [data-testid="stChatMessageContent"][aria-label*="from assistant"] span {
        color: #0f172a !important;   /* praticamente preto */
    }

    /* Avatar (custom :material/...:) */
    [data-testid="stChatMessageAvatarCustom"] {
        background: #0f172a !important;
        color: #ffffff !important;
    }
    [data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {
        color: #ffffff !important;
    }

    /* Links nas respostas */
    [data-testid="stChatMessageContent"] a {
        color: #1d4ed8 !important;
        font-weight: 600;
        text-decoration: none;
        border-bottom: 1px solid rgba(29, 78, 216, 0.35);
    }
    [data-testid="stChatMessageContent"] a:hover {
        border-bottom-color: #1d4ed8;
    }

    /* ================================================================
       SIDEBAR
       ================================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;     /* contraste reforçado */
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .material-symbols-rounded {
        color: #60a5fa;
        font-size: 1.15rem;
        margin-right: 0.4rem;
    }
    [data-testid="stSidebar"] code {
        color: #bfdbfe !important;
        background: rgba(96, 165, 250, 0.18) !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(96, 165, 250, 0.08) !important;
        border: 1px solid rgba(96, 165, 250, 0.25) !important;
    }
    /* Título do expander (summary) precisa de override explícito no fundo escuro */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
        color: #e2e8f0 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] li {
        color: #cbd5e1 !important;
        font-size: 0.83rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12) !important;
    }

    /* ================================================================
       INPUT DO CHAT  (corrigido para o DOM do Streamlit 1.5x)
       ================================================================ */
    /* Faixa inferior acompanha o fundo da app */
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    [data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto !important;
        border-radius: 16px !important;
        border: 2px solid #cbd5e1 !important;
        background: #ffffff !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 1rem !important;
        color: #0f172a !important;
        font-weight: 500;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    /* Botão de envio: ícone branco sobre fundo azul quando ativo */
    [data-testid="stChatInputSubmitButton"] button {
        border-radius: 10px !important;
        transition: opacity 0.2s ease !important;
    }
    [data-testid="stChatInputSubmitButton"] button:not(:disabled) {
        background: #2563eb !important;
    }
    [data-testid="stChatInputSubmitButton"] button:not(:disabled) span,
    [data-testid="stChatInputSubmitButton"] button:not(:disabled) svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stChatInputSubmitButton"] button:disabled {
        opacity: 0.35 !important;
    }

    /* ================================================================
       CARDS DE FONTES
       ================================================================ */
    .source-card {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #f1f5f9;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 9px 13px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #334155;
    }
    .source-card .material-symbols-rounded {
        font-size: 1.05rem;
        color: #2563eb;
    }

    /* ================================================================
       BOTÕES DE SUGESTÃO / AÇÃO (área principal)
       ================================================================ */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: 1.5px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 14px 18px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
    }
    .stButton > button:hover {
        border-color: #3b82f6 !important;
        color: #1d4ed8 !important;
        background: #f0f7ff !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.18) !important;
        transform: translateY(-2px);
    }
    .stButton > button p { font-weight: 600 !important; }

    /* Botão dentro da sidebar (Limpar conversa) — precisa de contraste no fundo escuro */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #f1f5f9 !important;
        border: 1.5px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button p {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.22) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        color: #ffffff !important;
        transform: none;
        box-shadow: none !important;
    }

    /* ================================================================
       EXPANDERS
       ================================================================ */
    [data-testid="stExpander"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
    }
    [data-testid="stExpander"] summary {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ================================================================
       SUGESTÕES — título
       ================================================================ */
    .suggestions-title {
        max-width: 820px;
        margin: 1.5rem auto 0.8rem auto;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
    }
    .suggestions-title .material-symbols-rounded {
        color: #f59e0b;
        font-size: 1.3rem;
        font-variation-settings: 'FILL' 1, 'wght' 500;
    }

    /* ================================================================
       SCROLLBAR
       ================================================================ */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* ================================================================
       RESPONSIVE — tablet (≤1024px)
       ================================================================ */
    @media (max-width: 1024px) {
        .main-header h1 { font-size: 2rem; }
        .chat-container { padding: 0.75rem; }
    }

    /* ================================================================
       RESPONSIVE — mobile (≤768px)
       ================================================================ */
    @media (max-width: 768px) {
        /* Cabeçalho */
        .main-header { padding: 1.2rem 0.75rem 0.4rem; }
        .main-header h1 {
            font-size: 1.5rem;
            letter-spacing: -0.02em;
        }
        .main-header h1 .material-symbols-rounded { font-size: 1.5rem; }
        .main-header .subtitle { font-size: 0.82rem; }

        /* Chat ocupa largura total */
        .chat-container { padding: 0.5rem 0.25rem; }
        [data-testid="stChatMessageContent"][aria-label*="from user"],
        [data-testid="stChatMessageContent"][aria-label*="from assistant"] {
            padding: 10px 14px !important;
            font-size: 0.9rem !important;
        }

        /* Input */
        [data-testid="stChatInput"] { border-radius: 12px !important; }
        [data-testid="stChatInput"] textarea { font-size: 0.92rem !important; }

        /* Botões de sugestão — tamanho menor */
        .stButton > button {
            font-size: 0.82rem !important;
            padding: 10px 12px !important;
        }

        /* Colunas de sugestão empilham em coluna única */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Título das sugestões */
        .suggestions-title {
            font-size: 0.92rem;
            margin: 1rem 0.25rem 0.5rem;
        }

        /* Expander de fontes */
        [data-testid="stExpander"] summary {
            font-size: 0.82rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Inicialização de estado ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "article_sources" not in st.session_state:
    st.session_state.article_sources = []
if "active_key" not in st.session_state:
    st.session_state.active_key = None  # chave em uso (detectar trocas)

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## :material/menu_book: Base de Conhecimento")
    st.markdown(":material/smart_toy: Modelo: `gpt-4o-mini`")
    st.markdown(":material/hub: Embeddings: `text-embedding-3-small`")

    st.markdown("---")

    # Chave de API do usuário
    st.markdown(":material/key: **Sua chave OpenAI** *(opcional)*")
    user_key_input = st.text_input(
        "Chave OpenAI",
        type="password",
        placeholder="sk-... (deixe vazio para usar a padrão)",
        label_visibility="collapsed",
        key="user_api_key_input"
    )
    user_key = user_key_input.strip()
    effective_key = user_key if user_key.startswith("sk-") else API_KEY

    # Invalida a chain se a chave mudou
    if effective_key != st.session_state.active_key:
        st.session_state.qa_chain = None
        st.session_state.active_key = effective_key

    st.markdown("---")

    # Lista de artigos indexados
    if st.session_state.article_sources:
        with st.expander(
            f"Artigos indexados ({len(st.session_state.article_sources)})",
            icon=":material/description:"
        ):
            for art in st.session_state.article_sources:
                st.markdown(f"- {art}")
    else:
        st.markdown(":material/description: Carregando artigos...")

    st.markdown("---")
    with st.expander("Sobre este Chatbot", icon=":material/info:"):
        st.markdown("""
        **Assistente acadêmico** especializado em
        comportamento do consumidor.

        Utiliza **RAG** (*Retrieval-Augmented Generation*)
        para buscar trechos relevantes dos artigos
        e responder com embasamento científico.
        """)

    st.markdown("---")
    if st.button("Limpar conversa", icon=":material/delete:", use_container_width=True):
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.rerun()

# ── Cabeçalho ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1><span class="material-symbols-rounded">psychology</span> Comportamento do Consumidor</h1>
    <p class="subtitle">Converse com a literatura acadêmica • 43 artigos indexados via RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Inicializar QA Chain ────────────────────────────────────────────────
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

if not os.path.exists(VECTORSTORE_DIR):
    st.error("Vectorstore não encontrada! Execute `python ingest.py` primeiro.",
             icon=":material/warning:")
    st.stop()

if st.session_state.qa_chain is None:
    try:
        with st.spinner("Conectando à base de conhecimento..."):
            embeddings = get_embeddings(effective_key)
            vectorstore = load_vectorstore(VECTORSTORE_DIR, embeddings)
            llm = get_llm(effective_key)
            st.session_state.qa_chain = create_qa_chain(vectorstore, llm)
            # Extrai nomes dos artigos para exibir na sidebar
            if not st.session_state.article_sources:
                def _year_key(name: str) -> int:
                    m = re.search(r'\b(19|20)\d{2}\b', name)
                    return int(m.group()) if m else 9999

                st.session_state.article_sources = sorted(
                    set(
                        doc.metadata.get("source", "")
                        for doc in vectorstore.docstore._dict.values()
                        if doc.metadata.get("source")
                    ),
                    key=_year_key
                )
                st.rerun()
    except Exception as e:
        st.error(f"Erro ao conectar: {e}")
        st.stop()

# ── Avatares (ícones Material Symbols, nativos do Streamlit) ────────────
USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/smart_toy:"


def render_sources(sources):
    """Renderiza as fontes consultadas em cartões com ícone."""
    with st.expander("Fontes consultadas", icon=":material/menu_book:"):
        for src in sources:
            st.markdown(
                f'<div class="source-card">'
                f'<span class="material-symbols-rounded">description</span>{src}</div>',
                unsafe_allow_html=True
            )


# ── Área do Chat ────────────────────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Captura nova pergunta digitada
if prompt := st.chat_input("Pergunte sobre os artigos..."):
    if st.session_state.qa_chain is None:
        st.warning("Aguarde a conexão com a base de conhecimento.",
                   icon=":material/hourglass_empty:")
        st.stop()
    st.session_state.messages.append({
        "role": "user",
        "avatar": USER_AVATAR,
        "content": prompt
    })
    st.rerun()

# Gera resposta sempre que a última mensagem for do usuário
# (vale tanto para perguntas digitadas quanto para os botões de sugestão)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    pergunta = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Analisando os artigos..."):
            try:
                # Passa todo o histórico anterior (exceto a última msg do user, que é a pergunta atual)
                history_before = st.session_state.messages[:-1]
                result = answer_question(
                    st.session_state.qa_chain,
                    pergunta,
                    chat_history=history_before
                )
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Erro ao gerar resposta: {str(e)}"
                sources = []
        st.markdown(answer)
        if sources:
            render_sources(sources)
    st.session_state.messages.append({
        "role": "assistant",
        "avatar": BOT_AVATAR,
        "content": answer,
        "sources": sources
    })
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
