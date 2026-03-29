class Artifact:
    def __init__(self, id):
        self.id = id
        self.stable_frames = 0
        self.current_cell = None
        self.confirmed = False

    def update(self, cell):
        if cell == self.current_cell:
            self.stable_frames += 1
        else:
            self.current_cell = cell
            self.stable_frames = 1

        if self.stable_frames >= 15:  # ~0.5 sec @30fps
            self.confirmed = True