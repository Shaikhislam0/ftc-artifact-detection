from typing import List

class Ramp:
    def __init__(self):
        self.state = ["NONE"] * 9

    def update_cell(self, idx: int, value: str):
        if idx < 0 or idx >= 9:
            return
        self.state[idx] = value

    def reset(self):
        self.state = ["NONE"] * 9


class Pattern:
    def __init__(self, motif: List[str]):
        assert len(motif) == 3
        self.pattern = motif * 3

    def score(self, ramp_state: List[str]) -> int:
        score = 0
        for i in range(9):
            if ramp_state[i] == self.pattern[i]:
                score += 2
        return score
