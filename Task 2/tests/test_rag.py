import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from config import ROOT, PDF, ABSTAIN, api_key
from generation import Generator, validate_answer, normalize_citations
from ingestion import load_index, chunk_document, get_model
from pipeline import RAGPipeline
from retrieval import Retriever, evidence_relevance_check

HISTORY = ROOT / 'evaluation'

@pytest.fixture(scope='session')
def rag():
    return RAGPipeline()

def test_initialization():
    import streamlit
    import fitz
    assert api_key()
    assert Generator().client
    chunks, vectors = load_index()
    assert len(chunks) == 212
    assert vectors.shape == (212, 384)
    with fitz.open(PDF) as doc:
        assert len(doc) == 48

def test_rebuild_chunks(rag):
    import fitz
    with fitz.open(PDF) as doc:
        pages = [{'page':i, 'text':p.get_text('text').strip()} for i,p in enumerate(doc,1)]
    assert chunk_document(pages,128,24) == rag.retriever.chunks

def test_retrieval_regression(rag):
    historical = pd.read_csv(HISTORY / 'hybrid_top5_details.csv')
    questions = pd.read_csv(ROOT / 'evaluation/questions_15.csv')
    mismatches = []
    for row in questions.itertuples():
        actual = rag.retriever.hybrid_retrieve(row.question)
        expected = historical[(historical.question_id == row.question_id) & (historical.chunk_set == 'small')].sort_values('rank')
        if [x['text'] for x in actual] != expected.retrieved_text.tolist():
            mismatches.append(row.question_id)
        assert all(x['chunk_id'] in {c['chunk_id'] for c in rag.retriever.chunks} for x in actual)
        assert all(1 <= x['page'] <= 48 for x in actual)
    assert not mismatches, f'Historical ranking differences: {mismatches}'

def test_query_bm25(rag):
    vector = rag.retriever.model.encode(['MEASURE function'], normalize_embeddings=True)[0]
    assert vector.shape == (384,)
    assert np.isclose(np.linalg.norm(vector), 1)
    scores = rag.retriever.bm25.score('MEASURE function')
    assert scores.shape == (212,) and scores.max() > 0

def test_evidence_regression(rag):
    records = json.loads((HISTORY/'groq_gpt_oss_20b_15.json').read_text(encoding='utf-8'))
    differences = []
    for record in records:
        actual = rag.retriever.select_complete_evidence(record['question'])
        if [(e['page'],e['text']) for e in actual] != [(e['page'],e['text']) for e in record['evidence']]:
            differences.append(record['question_id'])
    assert not differences, f'Focused evidence differs from Groq experiment: {differences}'

@pytest.mark.parametrize('question', [
    'What are the four core functions of the AI RMF?',
    'What is the purpose of the MEASURE function?',
    'What is the difference between explainability and interpretability?'])
def test_evidence(rag, question):
    evidence = rag.retriever.select_complete_evidence(question)
    assert len(evidence) == 3
    assert len({x['chunk_id'] for x in evidence}) == 3
    assert all(len(re.findall(r'\b\w+\b',x['text'])) >= 12 for x in evidence)
    chunks = {x['chunk_id']:x for x in rag.retriever.chunks}
    assert all(chunks[x['chunk_id']]['page']==x['page'] for x in evidence)

@pytest.mark.parametrize('case', json.loads((HISTORY/'groq_unsupported_7.json').read_text())[:6], ids=lambda c:c['question_id'])
def test_gate_no_api(rag, case, monkeypatch):
    def forbidden(*args):
        pytest.fail('Gate rejection called Groq')
    monkeypatch.setattr(rag.generator,'generate',forbidden)
    result = rag.ask(case['question'])
    assert result['answer'] == ABSTAIN
    assert not result['llm_called']
    assert result['abstention_reason'] == 'evidence_gate'

def test_verification():
    evidence = [dict(id='E1',page=33,text='The MEASURE function uses qualitative tools to analyze AI risks.')]
    assert normalize_citations('Tools 【E1】') == 'Tools [E1]'
    assert not validate_answer('It uses tools [E1]', evidence)['abstained']
    assert validate_answer('Tools [E9]',evidence)['abstention_reason']=='invalid_or_missing_citation'
    assert validate_answer('A $50,000 fine [E1]',evidence)['abstention_reason']=='unsupported_numerical_claim'

def test_security():
    assert '.env' in (ROOT/'.gitignore').read_text().splitlines()
    for path in ROOT.glob('*.py'):
        text=path.read_text(encoding='utf-8')
        assert api_key() not in text
        assert 'gsk_' not in text
        assert not any(x in text.lower() for x in ['langchain', 'llamaindex', 'qwen'])

def test_chatbot_integration():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=180).run()
    assert not app.exception
    assert not app.error
    assert 'Index loaded' in ' '.join(x.value for x in app.caption)
    app.chat_input[0].set_value('Who won the 2022 FIFA World Cup?').run()
    assert not app.exception
    assert app.session_state['history'][0]['answer'] == ABSTAIN

@pytest.mark.live
@pytest.mark.skipif(os.getenv('RUN_GROQ_TESTS') != '1', reason='Set RUN_GROQ_TESTS=1 for real API tests')
@pytest.mark.parametrize('question,abstain', [
    ('What is the purpose of the MEASURE function?',False),
    ('What are the four core functions of the AI RMF?',False),
    ('What is the difference between explainability and interpretability?',False),
    (json.loads((HISTORY/'groq_unsupported_7.json').read_text())[-1]['question'],True)])
def test_live(rag,question,abstain):
    result = rag.ask(question)
    print(json.dumps({k:result.get(k) for k in ['question','answer','abstained','abstention_reason','llm_called','usage']}))
    assert result['llm_called']
    assert result['abstained'] == abstain
    assert result['usage']['total_tokens'] > 0
    if not abstain:
        assert result['cited_passages']
        assert result['citation_format_pass']
