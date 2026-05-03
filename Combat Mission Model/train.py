"""
Produce the deployable model artifact + versioned metadata for downstream integration.

Primary outputs:
  - Joblib bundle (trained scorer + profiles + team heuristic): ``MODEL_PATH``
  - JSON sidecar for CI/registry and non-Python consumers: ``MODEL_METADATA_PATH``

A separate *Athena* layer may consume predictions later (coaching, commander narrative);
that product is not required to ship this model.
"""
from model import MODEL_PATH, Athena


def main() -> None:
    print("Training mission performance model...")
    athena = Athena()
    metrics = athena.train()
    athena.save(MODEL_PATH)
    meta_path = athena.save_metadata()
    print(f"Artifact:  {MODEL_PATH}")
    print(f"Metadata: {meta_path}")
    print(f"CV MAE:    {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")


if __name__ == "__main__":
    main()
