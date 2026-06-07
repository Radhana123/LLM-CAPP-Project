# ===========================
# Feature Tokenizer — CAPP
# ===========================

MATERIAL_TOKENS = {
    "Aluminum":     201,
    "Steel":        202,
    "Titanium":     203,
    "Brass":        204,
    "Plastic":      205,
    "Carbon_Steel": 208,
    "Nickel":       209,
}

FEATURE_TOKENS = {
    "Hole":         101,
    "Slot":         102,
    "Flat_Surface": 103,
    "Thread":       104,
    "Pocket":       105,
    "Chamfer":      106,
    "Groove":       107,
}

TOLERANCE_TOKENS = {
    "0.005mm": 300,
    "0.01mm":  301,
    "0.02mm":  302,
    "0.05mm":  303,
    "0.1mm":   304,
}

OPERATION_TOKENS = {
    "Facing":          401,
    "Center_Drilling": 402,
    "Drilling":        403,
    "Reaming":         404,
    "Boring":          405,
    "Milling":         406,
    "Turning":         407,
    "Threading":       408,
    "Grinding":        409,
    "Inspection":      410,
    "Deburring":       411,
    "Chamfering":      412,
}

SPECIAL_TOKENS = {
    "PAD": 0,
    "UNK": 999,
    "SOS": 998,
    "EOS": 997,
}


class FeatureTokenizer:

    def __init__(self):
        self.vocab = {
            **MATERIAL_TOKENS,
            **FEATURE_TOKENS,
            **TOLERANCE_TOKENS,
            **OPERATION_TOKENS,
            **SPECIAL_TOKENS,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.PAD = 0
        self.UNK = 999

    def tokenize_part(self, part):
        material_token  = self.vocab.get(part["material"], self.UNK)
        feature_tokens  = [self.vocab.get(f, self.UNK) for f in part["features"]]
        tolerance_token = self.vocab.get(part["tolerance"], self.UNK)
        flat_sequence   = [material_token] + feature_tokens + [tolerance_token]
        return {
            "part_id":        part["part_id"],
            "material_token":  material_token,
            "feature_tokens":  feature_tokens,
            "tolerance_token": tolerance_token,
            "batch_size":      part["batch_size"],
            "flat_sequence":   flat_sequence,
        }

    def tokenize_operations(self, operations):
        return [self.vocab.get(op, self.UNK) for op in operations]

    def detokenize_operations(self, token_ids):
        return [self.id_to_token.get(tid, "UNKNOWN") for tid in token_ids]

    def pad_sequence(self, seq, max_len):
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [self.PAD] * (max_len - len(seq))


if __name__ == "__main__":
    tok = FeatureTokenizer()
    part = {
        "part_id":    "P001",
        "material":   "Aluminum",
        "features":   ["Hole", "Slot"],
        "tolerance":  "0.02mm",
        "batch_size": 500,
    }
    result = tok.tokenize_part(part)
    print("Tokenized:", result)
    print("Flat sequence:", result["flat_sequence"])
    print("Vocab size:", len(tok.vocab))

    ops = ["Facing", "Drilling", "Inspection"]
    tokens = tok.tokenize_operations(ops)
    back   = tok.detokenize_operations(tokens)
    print("Ops:", ops, "-> Tokens:", tokens, "-> Back:", back)

    padded = tok.pad_sequence([201, 101], 8)
    print("Padded:", padded)