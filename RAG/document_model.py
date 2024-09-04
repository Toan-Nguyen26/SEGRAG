from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class QAEntry:
    question: str
    context: str
    answers: List[str]

@dataclass
class DocumentEntry:
    id: int
    title: str
    qas: List[QAEntry] = field(default_factory=list)
    content: List[str] = field(default_factory=list)
    num_sentences: int = 0
    segmented_sentences: List[str] = field(default_factory=list)