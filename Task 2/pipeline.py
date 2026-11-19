from config import ABSTAIN, TOP_K, CANDIDATE_K, RRF_K, MIN_COVERAGE
from retrieval import Retriever, evidence_relevance_check
from generation import Generator

class RAGPipeline:
    def __init__(self):
        self.generator = Generator()
        self.retriever = Retriever()

    def ask(self, question):
        if not question.strip():
            raise ValueError('Enter a question.')
        sources = self.retriever.hybrid_retrieve(question, top_k=TOP_K, candidate_k=CANDIDATE_K, rrf_k=RRF_K)
        relevance = evidence_relevance_check(question, sources, MIN_COVERAGE)
        result = dict(question=question, sources=sources, source_pages=sorted({s['page'] for s in sources}),
                      relevance=relevance, retrieval_method='small_hybrid_rrf', evidence=[], cited_passages=[],
                      answer=ABSTAIN, abstained=True, abstention_reason='evidence_gate', llm_called=False, usage={})
        if not relevance['relevant']:
            return result
        evidence = self.retriever.select_complete_evidence(question, top_n=3)
        evidence = [dict(item, id=f'E{i}') for i,item in enumerate(evidence,1)]
        result['evidence'] = evidence
        if not evidence:
            result['abstention_reason'] = 'no_evidence'
            return result
        result.update(self.generator.generate(question, evidence))
        return result
