from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class EvalResult(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

@dataclass
class SubQuestion:
    text: str
    depends_on: Optional[List[int]] = None 

@dataclass
class AnswerResult:
    relevance: str
    answer_span: Optional[str]

@dataclass
class EvalMetrics:
    semantic_coherence: float
    answerability: float
    overall_assessment: float
    response_validity: bool
    
    @classmethod
    def from_llm_output(
        cls, 
        coherence: float, 
        answerability: float,
        overall: Optional[float] = None,
        validity: Optional[bool] = None
    ):
        if overall is None:
            overall = (coherence + answerability / 10) / 2
        
        if validity is None:
            validity = overall >= 6.0
        
        return cls(
            semantic_coherence=coherence,
            answerability=answerability,
            overall_assessment=overall,
            response_validity=validity
        )
    
    @classmethod
    def default(cls):
        return cls(
            semantic_coherence=5.0,
            answerability=50.0,
            overall_assessment=5.0,
            response_validity=False
        )

    def to_dict(self) -> dict:
        return {
            "semantic_coherence": self.semantic_coherence,
            "answerability": self.answerability,
            "overall_assessment": self.overall_assessment,
            "response_validity": self.response_validity
        }

@dataclass
class Document:
    title: str
    content: str
    url: str
    doc_id: str
    page_id: str
    section: str
    type_: str
    chunk_id: str
    sub_chunk_id: str
    total_sub_chunk: str
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "doc_id": self.doc_id,
            "page_id": self.page_id,
            "section": self.section,
            "type": self.type_,
            "chunk_id": self.chunk_id,
            "sub_chunk_id": self.sub_chunk_id,
            "total_sub_chunk": self.total_sub_chunk,
            "score": self.score
        }
