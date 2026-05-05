"""Alpha Reversal Model — label_alpha_reversal_m (aylık reversal binary hedef)."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("train", Path(__file__).parent / "train.py")
train_mod = importlib.util.module_from_spec(spec)
sys.modules["train"] = train_mod
spec.loader.exec_module(train_mod)

if __name__ == "__main__":
    train_mod.main(
        target="label_alpha_reversal_m",
        save_name="ensemble_artifacts_alpha_reversal.pkl",
        classification=True,
    )
