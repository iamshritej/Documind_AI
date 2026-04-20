from typing import Dict, List


class ChatMemory:
    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def clear(self) -> None:
        self.history = []

    def get(self) -> List[Dict[str, str]]:
        return self.history
