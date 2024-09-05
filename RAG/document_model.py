from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class QAEntry:
    question: str
    context: str
    answers: List[str]

    def to_dict(self):
        return {
            'question': self.question,
            'context': self.context,
            'answers': self.answers
        }

@dataclass
class DocumentEntry:
    id: int
    title: str
    qas: List[QAEntry] = field(default_factory=list)
    content: List[str] = field(default_factory=list)
    num_sentences: int = 0
    segmented_sentences: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'qas': [qa.to_dict() for qa in self.qas],
            'content': self.content,
            'num_sentences': self.num_sentences,
            'segmented_sentences': self.segmented_sentences
        }