import streamlit as st
from pipeline import RAGPipeline
st.set_page_config(page_title='NIST AI RMF — RAG Chatbot', page_icon='📖')
st.title('NIST AI RMF — RAG Chatbot')
st.caption('Ask about the 48-page NIST AI Risk Management Framework. MiniLM + BM25 hybrid retrieval, focused evidence, and Groq GPT-OSS-20B.')
@st.cache_resource
def load_pipeline():
    return RAGPipeline()
try:
    rag = load_pipeline()
except Exception as exc:
    st.error(str(exc))
    st.stop()
st.caption(f'Index loaded · {len(rag.retriever.chunks)} passages · PDF page numbers')
if 'history' not in st.session_state:
    st.session_state.history = []
def show_result(result):
    st.markdown(result['answer'])
    for passage in result['cited_passages']:
        st.caption(f"[{passage['id']}] PDF page {passage['page']}")
    if result['evidence']:
        with st.expander('Retrieved evidence'):
            for passage in result['evidence']:
                st.markdown(f"**[{passage['id']}] PDF page {passage['page']}**")
                st.write(passage['text'])
    if result['abstained']:
        st.caption('The supplied evidence did not support an answer.')
for turn in st.session_state.history:
    with st.chat_message('user'):
        st.write(turn['question'])
    with st.chat_message('assistant'):
        show_result(turn)
if question := st.chat_input('Ask a question about NIST AI RMF'):
    with st.chat_message('user'):
        st.write(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Finding evidence…'):
                result = rag.ask(question)
            show_result(result)
            st.session_state.history.append(result)
        except Exception as exc:
            st.error(str(exc))
