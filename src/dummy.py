import json
import random

# just return a random prediction for testing
verdict = random.choice(["real", "deepfake"])
probability = round(random.uniform(0.5, 1.0), 4)

result = {
    "verdict": verdict,
    "probability": probability
}

print(json.dumps(result))
