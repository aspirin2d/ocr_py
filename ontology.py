from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

from abc import ABC, abstractmethod

class Ontology(ABC):
    @abstractmethod
    def __init__(self):
        pass

@dataclass
class DetectionOntology(Ontology):
    promptMap: List[Tuple[Any, str]]

    def prompts(self) -> List[Any]:
        return [prompt for prompt, _ in self.promptMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.promptMap]

    def promptToClass(self, prompt: Any) -> str:
        for p, cls in self.promptMap:
            if p == prompt:
                return cls
        raise ValueError("Prompt not found in ontology")

    def classToPrompt(self, cls: str) -> Any:
        for p, c in self.promptMap:
            if c == cls:
                return p
        raise ValueError("Class not found in ontology")

@dataclass
class CaptionOntology(DetectionOntology):
    promptMap: List[Tuple[str, str]]

    def __init__(self, ontology: Dict[str, str]):
        self.promptMap = [(k, v) for k, v in ontology.items()]

        if len(self.promptMap) == 0:
            raise ValueError("Ontology is empty")

    def prompts(self) -> List[str]:
        return super().prompts()

    def classToPrompt(self, cls: str) -> str:
        return super().classToPrompt(cls)
