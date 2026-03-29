from game_state import Ramp, Pattern
from ramp_slots import RampSlots

ramp = Ramp()
pattern = Pattern(["G", "P", "P"])
slots = RampSlots()

slots.add_slot(0, [(0,0),(100,0),(100,100),(0,100)])
slots.add_slot(1, [(100,0),(200,0),(200,100),(100,100)])
slots.add_slot(2, [(200,0),(300,0),(300,100),(200,100)])

x, y = 150, 50
slot = slots.get_slot(x, y)

if slot is not None:
    ramp.update_cell(slot, "P")

print("Ramp state:", ramp.state)
print("Pattern score:", pattern.score(ramp.state))
