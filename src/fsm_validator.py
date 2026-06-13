VALID_TRANSITIONS = {
    "START":            ["Facing", "Turning", "Milling"],
    "Facing":           ["Center_Drilling", "Drilling", "Milling"],
    "Center_Drilling":  ["Drilling"],
    "Drilling":         ["Reaming", "Boring", "Threading", "Inspection", "Deburring"],
    "Reaming":          ["Boring", "Inspection", "Deburring"],
    "Boring":           ["Inspection", "Deburring"],
    "Milling":          ["Drilling", "Chamfering", "Deburring", "Inspection"],
    "Turning":          ["Threading", "Chamfering", "Inspection"],
    "Threading":        ["Chamfering", "Inspection", "Deburring"],
    "Chamfering":       ["Inspection", "Deburring"],
    "Grinding":         ["Inspection", "Deburring"],
    "Deburring":        ["Inspection"],
    "Inspection":       ["END"],
}

TERMINAL_STATES = ["Inspection"]


class FSMValidator:

    def __init__(self):
        self.transitions = VALID_TRANSITIONS
        self.terminal = TERMINAL_STATES

    def validate(self, operations):
        if not operations:
            return False, "Empty sequence"
        current = "START"
        for op in operations:
            allowed = self.transitions.get(current, [])
            if op not in allowed:
                return False, f"{current} ke baad {op} nahi ho sakta. Allowed: {allowed}"
            current = op
        if current not in self.terminal:
            return False, f"Sequence '{current}' pe khatam nahi ho sakti."
        return True, "Valid sequence!"

    def validate_and_fix(self, operations):
        is_valid, msg = self.validate(operations)
        if is_valid:
            return operations, "Already valid"
        if operations[-1] != "Inspection":
            fixed = operations + ["Inspection"]
            is_valid2, msg2 = self.validate(fixed)
            if is_valid2:
                return fixed, "Fixed: Inspection add kiya end mein"
        return operations, f"Could not fix: {msg}"


if __name__ == "__main__":
    fsm = FSMValidator()

    print("=" * 50)
    print("FSM VALIDATOR TEST")
    print("=" * 50)

    seq1 = ["Facing", "Drilling", "Inspection"]
    valid, msg = fsm.validate(seq1)
    print(f"\nTest 1 - {seq1}")
    print(f"  Result: {'VALID' if valid else 'INVALID'} — {msg}")

    seq2 = ["Drilling", "Facing", "Inspection"]
    valid, msg = fsm.validate(seq2)
    print(f"\nTest 2 - {seq2}")
    print(f"  Result: {'VALID' if valid else 'INVALID'} — {msg}")

    seq3 = ["Facing", "Drilling"]
    fixed, fix_msg = fsm.validate_and_fix(seq3)
    print(f"\nTest 3 - Fix: {seq3}")
    print(f"  Fixed: {fixed} — {fix_msg}")

    print("\nSab materials ke sequences test:")
    test_seqs = {
        "Aluminum": ["Facing", "Drilling", "Milling", "Inspection"],
        "Steel":    ["Facing", "Center_Drilling", "Drilling", "Reaming", "Inspection"],
        "Brass":    ["Turning", "Threading", "Chamfering", "Inspection"],
    }
    for mat, seq in test_seqs.items():
        v, m = fsm.validate(seq)
        print(f"  {mat:10}: {'OK' if v else 'FAIL'} — {m}")