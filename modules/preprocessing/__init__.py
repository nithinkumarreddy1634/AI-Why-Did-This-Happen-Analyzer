"""Preprocessing package exporting all data cleaning, transformation, and coordinator tools."""

from modules.preprocessing.transformation_history import (
    TransformationStep,
    TransformationHistory,
)
from modules.preprocessing.missing_values import (
    detect_missing_values,
    get_missing_summary,
    impute_numerical_column,
    impute_categorical_column,
    drop_missing_rows,
    drop_missing_columns,
)
from modules.preprocessing.duplicates import (
    detect_duplicates,
    remove_duplicates,
)
from modules.preprocessing.outliers import (
    get_numerical_columns,
    detect_column_outliers,
    detect_all_outliers,
    handle_column_outliers,
)
from modules.preprocessing.categorical_cleaning import (
    get_categorical_columns,
    detect_categorical_inconsistencies,
    strip_column_whitespace,
    standardize_capitalization,
    replace_category_values,
    group_rare_categories,
)
from modules.preprocessing.data_types import (
    inspect_data_types,
    convert_column_type,
)
from modules.preprocessing.encoding import (
    CategoricalEncoder,
    LabelEncodingTransformer,
    OneHotEncodingTransformer,
    OrdinalEncodingTransformer,
    is_likely_identifier,
    apply_encoding,
)
from modules.preprocessing.scaling import (
    FeatureScaler,
    StandardScalerWrapper,
    MinMaxScalerWrapper,
    RobustScalerWrapper,
    apply_scaling,
)
from modules.preprocessing.coordinator import (
    DataPreprocessor,
    Preprocessor,
    PreprocessingEngine,
)
from modules.preprocessing.comparison import (
    compute_comparison_metrics,
    build_missing_comparison_chart,
    build_outlier_comparison_chart,
    build_distribution_comparison_chart,
)
from modules.preprocessing.export import (
    export_processed_dataset,
)

__all__ = [
    "TransformationStep",
    "TransformationHistory",
    "detect_missing_values",
    "get_missing_summary",
    "impute_numerical_column",
    "impute_categorical_column",
    "drop_missing_rows",
    "drop_missing_columns",
    "detect_duplicates",
    "remove_duplicates",
    "get_numerical_columns",
    "detect_column_outliers",
    "detect_all_outliers",
    "handle_column_outliers",
    "get_categorical_columns",
    "detect_categorical_inconsistencies",
    "strip_column_whitespace",
    "standardize_capitalization",
    "replace_category_values",
    "group_rare_categories",
    "inspect_data_types",
    "convert_column_type",
    "CategoricalEncoder",
    "LabelEncodingTransformer",
    "OneHotEncodingTransformer",
    "OrdinalEncodingTransformer",
    "is_likely_identifier",
    "apply_encoding",
    "FeatureScaler",
    "StandardScalerWrapper",
    "MinMaxScalerWrapper",
    "RobustScalerWrapper",
    "apply_scaling",
    "DataPreprocessor",
    "Preprocessor",
    "PreprocessingEngine",
    "compute_comparison_metrics",
    "build_missing_comparison_chart",
    "build_outlier_comparison_chart",
    "build_distribution_comparison_chart",
    "export_processed_dataset",
]
