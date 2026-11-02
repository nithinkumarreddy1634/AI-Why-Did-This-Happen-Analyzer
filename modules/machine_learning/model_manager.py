"""Model serialization, metadata tracking, and model registry management."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
import time
import json
import uuid
import joblib
from io import BytesIO

from sklearn.pipeline import Pipeline
from modules.machine_learning.problem_detector import ProblemType


@dataclass
class ModelArtifact:
    """Complete bundle containing trained pipeline, metadata, and evaluation metrics."""
    model_id: str
    model_name: str
    problem_type: str
    pipeline: Pipeline
    target_name: str
    feature_names_in: List[str]
    feature_names_out: List[str]
    metrics: Dict[str, Any]
    target_classes: Optional[List[str]] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def get_summary(self) -> Dict[str, Any]:
        """Return metadata summary suitable for JSON serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "problem_type": self.problem_type,
            "target_name": self.target_name,
            "feature_names_in": self.feature_names_in,
            "n_features_in": len(self.feature_names_in),
            "n_features_out": len(self.feature_names_out),
            "target_classes": self.target_classes,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "hyperparameters": self.hyperparameters,
            "notes": self.notes,
        }


class ModelRegistry:
    """In-memory and session-level model artifact manager."""

    def __init__(self) -> None:
        self.models: Dict[str, ModelArtifact] = {}

    def register_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        problem_type: Union[ProblemType, str],
        target_name: str,
        feature_names_in: List[str],
        feature_names_out: List[str],
        metrics: Dict[str, Any],
        target_classes: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> ModelArtifact:
        """Register a newly trained model pipeline with complete audit metadata."""
        model_id = str(uuid.uuid4())[:8]
        prob_type_str = problem_type.value if isinstance(problem_type, ProblemType) else str(problem_type)

        artifact = ModelArtifact(
            model_id=model_id,
            model_name=model_name,
            problem_type=prob_type_str,
            pipeline=pipeline,
            target_name=target_name,
            feature_names_in=feature_names_in,
            feature_names_out=feature_names_out,
            metrics=metrics,
            target_classes=target_classes,
            hyperparameters=hyperparameters or {},
            notes=notes,
        )
        self.models[model_id] = artifact
        return artifact

    def get_model(self, model_id: str) -> Optional[ModelArtifact]:
        """Retrieve model artifact by id."""
        return self.models.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """List summary of all registered models."""
        return [art.get_summary() for art in self.models.values()]

    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry."""
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False

    def export_model_bytes(self, model_id: str) -> Optional[bytes]:
        """Export serialized artifact bytes for download."""
        artifact = self.models.get(model_id)
        if not artifact:
            return None
        buffer = BytesIO()
        joblib.dump(artifact, buffer)
        return buffer.getvalue()

    def import_model_bytes(self, data_bytes: bytes) -> ModelArtifact:
        """Import model artifact from user-uploaded bytes with strict validation."""
        buffer = BytesIO(data_bytes)
        artifact = joblib.load(buffer)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError("Uploaded file is not a valid ModelArtifact.")
        if not hasattr(artifact, "pipeline") or not isinstance(artifact.pipeline, Pipeline):
            raise ValueError("Artifact does not contain a valid scikit-learn Pipeline.")
        self.models[artifact.model_id] = artifact
        return artifact
