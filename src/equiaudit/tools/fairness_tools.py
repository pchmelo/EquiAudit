import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from equiaudit.tools.tool import Tool
from equiaudit.tools.tool_manager import ToolManager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from itertools import combinations as iter_combinations
import warnings
import os


warnings.simplefilter(action='ignore', category=Warning)

class FairnessTools(ToolManager):    
    def __init__(self):
        super().__init__()
        
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        self.tool_check_missing = Tool(
            name="check_missing_data",
            function=self.check_missing_data,
            description="Analyze missing data in the dataset",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_detect_sensitive = Tool(
            name="detect_sensitive_attributes",
            function=self.detect_sensitive_attributes,
            description="Detect sensitive/protected attributes in the dataset",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_check_imbalance = Tool(
            name="check_class_imbalance",
            function=self.check_class_imbalance,
            description="Check for class imbalance in categorical features",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_load_dataset = Tool(
            name="load_dataset",
            function=self.load_dataset,
            description="Load a CSV dataset from the data directory. Returns dataset info and preview.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset to load (with or without .csv extension)"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_get_dataset_preview = Tool(
            name="get_dataset_preview",
            function=self.get_dataset_preview,
            description="Get detailed preview of dataset including all column names, types, sample values, and statistics. Use this to understand the dataset structure before analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"}
                },
                "required": ["dataset_name"]
            }
        )
        
        self.tool_analyze_sensitive = Tool(
            name="analyze_sensitive_column",
            function=self.analyze_sensitive_column,
            description="Analyze a specific column for sensitive attributes and fairness concerns. Provides distribution and statistical analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "column_name": {"type": "string", "description": "Name of the column to analyze"}
                },
                "required": ["dataset_name", "column_name"]
            }
        )
        
        self.tool_fairness_analysis = Tool(
            name="analyze_target_fairness",
            function=self.analyze_target_fairness,
            description="Analyze fairness metrics for target variable across sensitive attributes. Generates visualizations and statistical analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "target_column": {"type": "string", "description": "Name of the target column"},
                    "sensitive_columns": {"type": "array", "items": {"type": "string"}, "description": "List of sensitive attribute column names"},
                    "output_dir": {"type": "string", "description": "Directory to save visualizations"}
                },
                "required": ["dataset_name", "target_column", "sensitive_columns", "output_dir"]
            }
        )

        self.tool_ml_model_analysis = Tool(
            name="train_and_evaluate_ml_model",
            function=self.train_and_evaluate_ml_model,
            description="Train a ml model to evaluate performance (F1 Score) and fairness metrics.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name of the dataset"},
                    "target_column": {"type": "string", "description": "Target variable"},
                    "sensitive_columns": {"type": "array", "items": {"type": "string"}, "description": "List of sensitive columns"},
                    "test_size": {"type": "number", "description": "Test set size fraction (default 0.25)"},
                    "model_type": {"type": "string", "description": "Model type (Random Forest, Logistic Regression, etc.)"},
                    "model_params": {"type": "object", "description": "Model hyperparameters"}
                },
                "required": ["dataset_name", "target_column"]
            }
        )
        
        self.list_of_tools = [
            self.tool_load_dataset,
            self.tool_get_dataset_preview,
            self.tool_check_missing,
            self.tool_detect_sensitive,
            self.tool_analyze_sensitive,
            self.tool_check_imbalance,
            self.tool_fairness_analysis,
            self.tool_ml_model_analysis
        ]
        self._build_tool_mappings()
    
    def load_dataset(self, dataset_name: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "path": path,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(3).to_dict(orient="records")
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_dataset_preview(self, dataset_name: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            column_info = []
            for col in df.columns:
                col_data = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().head(5).tolist()
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_data["stats"] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean())
                    }
                elif df[col].dtype == 'object':
                    top_values = df[col].value_counts().head(3)
                    col_data["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                
                column_info.append(col_data)
            
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_details": column_info
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def analyze_sensitive_column(self, dataset_name: str, column_name: str) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if column_name not in df.columns:
                return {"status": "error", "message": f"Column '{column_name}' not found"}
            
            col_data = df[column_name]
            result = {
                "status": "success",
                "column": column_name,
                "type": str(col_data.dtype),
                "unique_values": int(col_data.nunique()),
                "null_count": int(col_data.isnull().sum())
            }
            
            if col_data.dtype == 'object' or col_data.nunique() < 50:
                value_counts = col_data.value_counts()
                proportions = (value_counts / len(df) * 100).round(2)
                result["distribution"] = {str(k): {"count": int(v), "percentage": float(proportions[k])} 
                                         for k, v in value_counts.head(20).items()}
                result["imbalance_ratio"] = float(proportions.iloc[0] / proportions.iloc[-1]) if len(proportions) > 1 else 1.0
            else:
                result["stats"] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std())
                }
            
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _resolve_path(self, dataset_name: str) -> str:
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4]

        # When the GUI is launched with a user-supplied dataset, honour that path.
        env_dataset = os.environ.get("FAIRNESS_DATASET_PATH", "")
        if env_dataset and os.path.exists(env_dataset):
            env_stem = os.path.splitext(os.path.basename(env_dataset))[0]
            if dataset_name == env_stem or dataset_name == os.path.basename(env_dataset):
                return env_dataset

        possible_paths = [
            os.path.join(self.data_dir, f"{dataset_name}.csv"),
            os.path.join(self.data_dir, dataset_name), 
            dataset_name,  
            f"{dataset_name}.csv"  
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found. Tried:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths) +
            f"\n\nData directory: {self.data_dir}" +
            f"\n\nAvailable datasets: {self._list_available_datasets()}"
        )
    
    def _list_available_datasets(self) -> str:
        try:
            if os.path.exists(self.data_dir):
                files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
                return ", ".join(files) if files else "No CSV files found"
            return f"Data directory not found: {self.data_dir}"
        except Exception as e:
            return f"Unable to list datasets: {str(e)}"
    
    
    def _detect_type_inconsistencies(self, series):
        inconsistencies = {
            "has_inconsistency": False,
            "numeric_count": 0,
            "string_count": 0,
            "mixed_type_values": [],
            "suspicious_values": []
        }
        
        if series.dtype == 'object':
            numeric_mask = series.apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and str(x).replace('.','',1).replace('-','',1).isdigit()))
            string_mask = series.apply(lambda x: isinstance(x, str) and not str(x).replace('.','',1).replace('-','',1).isdigit())
            
            inconsistencies["numeric_count"] = int(numeric_mask.sum())
            inconsistencies["string_count"] = int(string_mask.sum())
            
            if inconsistencies["numeric_count"] > 0 and inconsistencies["string_count"] > 0:
                inconsistencies["has_inconsistency"] = True
                
                string_samples = series[string_mask].unique()[:5].tolist()
                inconsistencies["mixed_type_values"] = [str(v) for v in string_samples]
        
        return inconsistencies
    
    def _detect_suspicious_patterns(self, series):
        suspicious = []
        
        if series.dtype == 'object':
            single_char = series[series.apply(lambda x: isinstance(x, str) and len(str(x).strip()) == 1)]
            if len(single_char) > 0:
                unique_chars = single_char.unique()
                for char in unique_chars:
                    count = (series == char).sum()
                    if count > 1: 
                        suspicious.append({
                            "pattern": f"Single character '{char}'",
                            "count": int(count),
                            "percentage": float(count / len(series) * 100)
                        })
            
            whitespace = series.apply(lambda x: isinstance(x, str) and len(str(x).strip()) == 0)
            if whitespace.sum() > 0:
                suspicious.append({
                    "pattern": "Whitespace/empty strings",
                    "count": int(whitespace.sum()),
                    "percentage": float(whitespace.sum() / len(series) * 100)
                })
        
        elif series.dtype in ['int64', 'float64']:
            suspicious_numbers = [-999, -99, -9, 999, 9999, 99999, -1]
            for num in suspicious_numbers:
                count = (series == num).sum()
                if count > 0:
                    suspicious.append({
                        "pattern": f"Suspicious numeric value {num}",
                        "count": int(count),
                        "percentage": float(count / len(series) * 100)
                    })
        
        return suspicious
    
    def check_missing_data(self, dataset_name: str):
        try:
            path = self._resolve_path(dataset_name)
            
            na_values = ['?', 'NA', 'N/A', 'n/a', 'na', 'NULL', 'null', 'None', 'none', 
                        '', ' ', 'NaN', 'nan', '--', '..', 'missing', 'Missing', 
                        'unknown', 'Unknown', 'UNKNOWN', 'undefined', 'Undefined']
            
            df_original = pd.read_csv(path, keep_default_na=False)
            
            df = pd.read_csv(path, na_values=na_values, keep_default_na=True)
            
            na_values_found = {}
            for col in df.columns:
                found_na = set()
                if df[col].isnull().sum() > 0:
                    mask = df[col].isnull()
                    original_values = df_original.loc[mask, col].unique()
                    found_na.update([str(v) for v in original_values if v != ''])
                na_values_found[col] = list(found_na)
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
            
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            
            result = {
                "status": "success",
                "dataset": dataset_name,
                "total_rows": len(df),
                "total_missing_values": int(missing_data.sum()),
                "overall_missing_percentage": float((missing_data.sum() / (len(df) * len(df.columns)) * 100)),
                "columns_with_issues": 0,
                "details": []
            }
            
            for col in df.columns:
                col_issues = {
                    "column": col,
                    "data_type": str(df[col].dtype),
                    "missing_count": int(missing_data[col]),
                    "missing_percentage": float(missing_pct[col])
                }
                
                issues = []
                
                inconsistencies = self._detect_type_inconsistencies(df[col].dropna())
                if inconsistencies["has_inconsistency"]:
                    col_issues["type_inconsistency"] = {
                        "detected": True,
                        "numeric_values": inconsistencies["numeric_count"],
                        "string_values": inconsistencies["string_count"],
                        "sample_string_values": inconsistencies["mixed_type_values"]
                    }
                    issues.append(f"Mixed types: {inconsistencies['numeric_count']} numeric, {inconsistencies['string_count']} string")
                
                suspicious = self._detect_suspicious_patterns(df[col])
                if suspicious:
                    col_issues["suspicious_patterns"] = suspicious
                    for pattern in suspicious:
                        issues.append(f"{pattern['pattern']}: {pattern['count']} occurrences ({pattern['percentage']:.2f}%)")
                
                if col_issues["missing_count"] > 0:
                    na_found = na_values_found.get(col, [])
                    if na_found:
                        issues.append(f"Missing values: {col_issues['missing_count']} ({col_issues['missing_percentage']:.2f}%) - Detected as NA: {na_found}")
                    else:
                        issues.append(f"Missing values: {col_issues['missing_count']} ({col_issues['missing_percentage']:.2f}%)")
                
                if (col_issues["missing_count"] > 0 or 
                    inconsistencies["has_inconsistency"] or 
                    len(suspicious) > 0):
                    col_issues["detected_issues"] = " | ".join(issues)
                    if col_issues["missing_count"] > 0 and na_values_found.get(col):
                        col_issues["na_values_detected"] = na_values_found[col]
                    result["details"].append(col_issues)
                    result["columns_with_issues"] += 1
            
            return result
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def detect_sensitive_attributes(self, dataset_name: str):
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            column_analysis = []
            
            for col in df.columns:
                col_info = {
                    "column": col,
                    "type": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().unique()[:10].tolist()
                }
                
                if df[col].dtype == 'object' or df[col].nunique() < 50:
                    value_counts = df[col].value_counts()
                    proportions = (value_counts / len(df) * 100).round(2)
                    col_info["top_values"] = {str(k): float(v) for k, v in proportions.head(5).items()}
                
                column_analysis.append(col_info)
            
            return {
                "status": "success",
                "dataset": dataset_name,
                "total_columns": len(column_analysis),
                "columns": column_analysis
            }
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def check_class_imbalance(self, dataset_name: str, columns_to_check: list = None):
        try:
            path = self._resolve_path(dataset_name)
            df = pd.read_csv(path)
            
            if columns_to_check is not None:
                cols_to_evaluate = [c for c in columns_to_check if c in df.columns]
            else:
                cols_to_evaluate = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
            imbalances = []
            
            for col in cols_to_evaluate:
                value_counts = df[col].value_counts()
                proportions = (value_counts / len(df) * 100).round(2)
                
                if columns_to_check is not None or proportions.iloc[0] > 65:
                    imbalances.append({
                        "column": col,
                        "dominant_value": str(proportions.index[0]),
                        "dominant_percentage": float(proportions.iloc[0]),
                        "distribution": proportions.head(5).to_dict()
                    })
            
            return {
                "status": "success",
                "dataset": dataset_name,
                "imbalanced_columns": len(imbalances),
                "details": imbalances
            }
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def analyze_target_fairness(self, dataset_name: str, target_column: str, sensitive_columns: list, output_dir: str, selected_pairs: list = None) -> dict:
        try:
            path = self._resolve_path(dataset_name)
            
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
            else:
                 base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                 output_dir = os.path.join(base_dir, "reports", "fairness_images")
                 os.makedirs(output_dir, exist_ok=True)
            
            na_values = ['?', 'NA', 'N/A', 'n/a', 'na', 'NULL', 'null', 'None', 'none', 
                        '', ' ', 'NaN', 'nan', '--', '..', 'missing', 'Missing', 
                        'unknown', 'Unknown', 'UNKNOWN', 'undefined', 'Undefined']
            
            df = pd.read_csv(path, na_values=na_values, keep_default_na=True)
            df = df.dropna().reset_index(drop=True)
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            for col in sensitive_columns:
                if col not in df.columns:
                    return {"status": "error", "message": f"Sensitive column '{col}' not found"}
            
            os.makedirs(output_dir, exist_ok=True)
            
            result = {
                "status": "success",
                "dataset": dataset_name,
                "target_column": target_column,
                "sensitive_columns": sensitive_columns,
                "total_rows": len(df),
                "target_distribution": {},
                "group_proportions": {},
                "target_rates_by_group": {},
                "combined_analysis": {},
                "generated_images": []
            }
            
            target_counts = df[target_column].value_counts()
            target_pct = (target_counts / len(df) * 100).round(2)
            result["target_distribution"] = {
                "counts": target_counts.to_dict(),
                "percentages": target_pct.to_dict()
            }
            
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(data=df, x=target_column)
            plt.title(f"{target_column} Distribution", fontsize=14, pad=15)
            plt.xlabel(target_column, fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            img_path = os.path.join(output_dir, f"target_distribution.png")
            plt.savefig(img_path, bbox_inches='tight', dpi=100)
            plt.close()
            result["generated_images"].append(img_path)
            
            for sensitive_col in sensitive_columns:
                group_counts = df[sensitive_col].value_counts()
                group_pct = (group_counts / len(df) * 100).round(2)
                result["group_proportions"][sensitive_col] = {
                    "counts": group_counts.to_dict(),
                    "percentages": group_pct.to_dict()
                }
            
            n_cols = len(sensitive_columns)
            fig, axes = plt.subplots(1, n_cols, figsize=(10*n_cols, 6))
            if n_cols == 1:
                axes = [axes]
            
            for idx, sensitive_col in enumerate(sensitive_columns):
                n_unique = df[sensitive_col].nunique()
                
                sns.countplot(data=df, x=sensitive_col, ax=axes[idx])
                axes[idx].set_title(f"{sensitive_col} Distribution", fontsize=12, pad=15)
                axes[idx].set_xlabel(sensitive_col, fontsize=10)
                axes[idx].set_ylabel("Count", fontsize=10)
                
                if n_unique > 5:
                    axes[idx].tick_params(axis='x', rotation=90)
                    for label in axes[idx].get_xticklabels():
                        label.set_rotation(90)
                        label.set_ha('center')
                else:
                    axes[idx].tick_params(axis='x', rotation=45)
                    for label in axes[idx].get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha('right')
            
            plt.tight_layout()
            img_path = os.path.join(output_dir, f"sensitive_distributions.png")
            plt.savefig(img_path, bbox_inches='tight', dpi=100)
            plt.close()
            result["generated_images"].append(img_path)
            
            fig, axes = plt.subplots(1, n_cols, figsize=(10*n_cols, 6))
            if n_cols == 1:
                axes = [axes]
            
            for idx, sensitive_col in enumerate(sensitive_columns):
                n_unique = df[sensitive_col].nunique()
                
                sns.countplot(data=df, x=sensitive_col, hue=target_column, ax=axes[idx])
                axes[idx].set_title(f"{target_column} by {sensitive_col}", fontsize=12, pad=15)
                axes[idx].set_xlabel(sensitive_col, fontsize=10)
                axes[idx].set_ylabel("Count", fontsize=10)
                
                if n_unique > 5:
                    axes[idx].tick_params(axis='x', rotation=90)
                    for label in axes[idx].get_xticklabels():
                        label.set_rotation(90)
                        label.set_ha('center')
                else:
                    axes[idx].tick_params(axis='x', rotation=45)
                    for label in axes[idx].get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha('right')
                
                axes[idx].legend(title=target_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            img_path = os.path.join(output_dir, f"target_by_sensitive.png")
            plt.savefig(img_path, bbox_inches='tight', dpi=100)
            plt.close()
            result["generated_images"].append(img_path)
            
            for sensitive_col in sensitive_columns:
                group_target_rates = {}
                for group_value in df[sensitive_col].unique():
                    group_df = df[df[sensitive_col] == group_value]
                    target_dist = group_df[target_column].value_counts()
                    target_pct = (target_dist / len(group_df) * 100).round(2)
                    group_target_rates[str(group_value)] = {
                        "total_count": len(group_df),
                        "target_distribution": target_dist.to_dict(),
                        "target_percentages": target_pct.to_dict()
                    }
                result["target_rates_by_group"][sensitive_col] = group_target_rates
            
            if len(sensitive_columns) >= 2:
                # selected_pairs=None means use all pairs
                # selected_pairs=[] means user skipped pair analysis
                # selected_pairs=[...] means use only those pairs
                if selected_pairs is not None and len(selected_pairs) == 0:
                    print("User skipped intersectional pair analysis")
                    sensitive_pairs = []
                elif selected_pairs:
                    sensitive_pairs = [
                        (col1, col2) for col1, col2 in selected_pairs 
                        if col1 in df.columns and col2 in df.columns
                    ]
                    print(f"Generating combinations for {len(sensitive_pairs)} user-selected pairs")
                else:
                    sensitive_pairs = list(iter_combinations(sensitive_columns, 2))
                    print(f"Generating combinations for all {len(sensitive_pairs)} possible pairs")
                
                for col1, col2 in sensitive_pairs:
                    combined_col = f"{col1}_{col2}"
                    df[combined_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
                    
                    combined_counts = df[combined_col].value_counts()
                    combined_pct = (combined_counts / len(df) * 100).round(2)
                    
                    result["combined_analysis"][combined_col] = {
                        "counts": combined_counts.to_dict(),
                        "percentages": combined_pct.to_dict()
                    }
                    
                    combined_dir = os.path.join(output_dir, f"{col1}_{col2}_combinations")
                    os.makedirs(combined_dir, exist_ok=True)
                    
                    sorted_counts = combined_counts.sort_values(ascending=False)
                    max_count = sorted_counts.iloc[0]
                    
                    scale_groups = {
                        'high': sorted_counts[sorted_counts >= max_count * 0.1],  # >= 10% of max
                        'medium': sorted_counts[(sorted_counts >= max_count * 0.01) & (sorted_counts < max_count * 0.1)],  # 1-10% of max
                        'low': sorted_counts[sorted_counts < max_count * 0.01]  # < 1% of max
                    }
                    
                    scale_groups = {k: v for k, v in scale_groups.items() if len(v) > 0}
                    
                    for scale_name, group_data in scale_groups.items():
                        if len(group_data) == 0:
                            continue
                        
                        df_filtered = df[df[combined_col].isin(group_data.index)]
                        
                        if len(df_filtered) == 0:
                            continue
                        
                        n_categories = len(group_data)
                        fig_width = max(16, min(n_categories * 0.6, 24))
                        
                        fig, ax = plt.subplots(figsize=(fig_width, 10))
                        order = group_data.index.tolist()
                        sns.countplot(data=df_filtered, x=combined_col, hue=target_column, order=order, ax=ax)
                        
                        count_range = f"{int(group_data.min())}-{int(group_data.max())}"
                        title = f"Target: {target_column}\n"
                        title += f"Analyzed by: {col1} & {col2}\n"
                        title += f"Scale: {scale_name.upper()} (Count range: {count_range})"
                        ax.set_title(title, fontsize=16, pad=25, fontweight='bold')
                        
                        ax.set_xlabel(f"{col1} & {col2} Combinations", 
                                     fontsize=13, fontweight='bold', labelpad=10)
                        ax.set_ylabel("Count", fontsize=13, fontweight='bold')
                        
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
                        
                        positions = range(len(order))
                        for pos, category in zip(positions, order):
                            parts = category.split('_')
                            label_text = f"{parts[0]}\n{parts[1]}" if len(parts) == 2 else category
                            ax.text(pos, -max(ax.get_ylim()) * 0.15, label_text, 
                                   ha='center', va='top', fontsize=9, fontweight='bold')
                        
                        for container in ax.containers:
                            ax.bar_label(container, fontsize=8, padding=3, fontweight='bold')
                        
                        ax.legend(title=f"{target_column} Values", bbox_to_anchor=(1.02, 1), 
                                 loc='upper left', fontsize=11, title_fontsize=12, frameon=True, shadow=True)
                        
                        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                        ax.set_axisbelow(True)
                        
                        plt.subplots_adjust(bottom=0.15)
                        plt.tight_layout()
                        
                        img_path = os.path.join(combined_dir, f"{scale_name}_scale.png")
                        plt.savefig(img_path, bbox_inches='tight', dpi=150, facecolor='white')
                        plt.close()
                        result["generated_images"].append(img_path)
                    
                    individual_dir = os.path.join(combined_dir, "individual_combinations")
                    os.makedirs(individual_dir, exist_ok=True)
                    
                    top_combinations = sorted_counts.head(50)
                    
                    for combination in top_combinations.index:
                        combo_df = df[df[combined_col] == combination]
                        
                        if len(combo_df) == 0:
                            continue
                        
                        fig, ax = plt.subplots(figsize=(10, 7))
                        
                        target_dist = combo_df[target_column].value_counts()
                        colors = sns.color_palette("Set2", len(target_dist))
                        
                        bars = ax.bar(range(len(target_dist)), target_dist.values, color=colors)
                        ax.set_xticks(range(len(target_dist)))
                        ax.set_xticklabels(target_dist.index, fontsize=12, fontweight='bold')
                        
                        parts = combination.split('_')
                        title = f"Target Distribution: {target_column}\n"
                        title += f"{col1}: {parts[0]}\n"
                        title += f"{col2}: {parts[1] if len(parts) > 1 else 'N/A'}\n"
                        title += f"Total Count: {len(combo_df)}"
                        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
                        
                        ax.set_xlabel(f"{target_column} Values", fontsize=12, fontweight='bold')
                        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
                        
                        for bar in bars:
                            height = bar.get_height()
                            percentage = (height / len(combo_df) * 100)
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}\n({percentage:.1f}%)',
                                   ha='center', va='bottom', fontsize=10, fontweight='bold')
                        
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        ax.set_axisbelow(True)
                        
                        plt.tight_layout()
                        
                        safe_name = combination.replace('<', 'lt').replace('>', 'gt').replace(':', '-')
                        safe_name = safe_name.replace('"', '').replace('/', '-').replace('\\', '-')
                        safe_name = safe_name.replace('|', '-').replace('?', '').replace('*', '')
                        safe_name = safe_name.replace(' ', '_')
                        
                        import hashlib
                        
                        # Fix for Windows MAX_PATH (260 char limit): Truncate very long combo names
                        if len(safe_name) > 30:
                            hash_suffix = hashlib.md5(safe_name.encode()).hexdigest()[:6]
                            safe_name = safe_name[:25] + "_" + hash_suffix
                        
                        img_path = os.path.join(individual_dir, f"{safe_name}.png")
                        
                        try:
                            plt.savefig(img_path, bbox_inches='tight', dpi=120, facecolor='white')
                        except Exception as e:
                            # Final fallback: if path is still magically too long, drop it instead of crashing the pipeline
                            pass
                        finally:
                            plt.close()
                            
                        if os.path.exists(img_path):
                            result["generated_images"].append(img_path)
                    
                    combined_target_rates = {}
                    for group_value in df[combined_col].unique():
                        group_df = df[df[combined_col] == group_value]
                        target_dist = group_df[target_column].value_counts()
                        target_pct = (target_dist / len(group_df) * 100).round(2)
                        combined_target_rates[str(group_value)] = {
                            "total_count": len(group_df),
                            "target_distribution": target_dist.to_dict(),
                            "target_percentages": target_pct.to_dict()
                        }
                    result["combined_analysis"][combined_col]["target_rates"] = combined_target_rates
                    result["combined_analysis"][combined_col]["scale_groups"] = {
                        k: {"categories": v.index.tolist(), "count_range": f"{int(v.min())}-{int(v.max())}"} 
                        for k, v in scale_groups.items()
                    }
                    result["combined_analysis"][combined_col]["output_directory"] = combined_dir
            
            return result
            
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _eval_presplit(self, train_df, test_df, target_column, sensitive_columns,
                       test_size, model_type, model_params):
        """Evaluate a model on pre-split DataFrames (split-first protocol for mitigation eval)."""
        _train = train_df.copy().dropna(subset=[target_column])
        _test = test_df.copy().dropna(subset=[target_column])

        X_tr = _train.drop(columns=[target_column])
        y_train_raw = _train[target_column]
        X_te = _test.drop(columns=[target_column])
        y_test_raw = _test[target_column]

        X_test_raw = X_te.copy()

        w_train = None
        if 'sample_weight' in X_tr.columns:
            w_train = X_tr['sample_weight'].values
            X_tr = X_tr.drop(columns=['sample_weight'])
            X_test_raw = X_test_raw.drop(columns=['sample_weight'], errors='ignore')
        if 'sample_weight' in X_te.columns:
            X_te = X_te.drop(columns=['sample_weight'])

        X_train = X_tr.copy()
        X_test = X_te.copy()

        for col in X_train.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_train[col] = X_train[col].fillna("Missing").astype(str)
            X_train[col] = le.fit_transform(X_train[col])
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna("Missing").astype(str).map(
                    lambda v, _le=le: int(_le.transform([v])[0]) if v in _le.classes_ else 0
                )

        for col in X_train.select_dtypes(include=['int64', 'float64']).columns:
            med = float(X_train[col].median())
            X_train[col] = X_train[col].fillna(med)
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna(med)

        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train_raw.astype(str))
        y_test = np.array([
            int(le_target.transform([v])[0]) if v in le_target.classes_ else 0
            for v in y_test_raw.astype(str)
        ])

        positive_label_idx = 1 if len(le_target.classes_) > 1 else 0
        positive_class_name = le_target.classes_[positive_label_idx]

        if model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000, **model_params)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42, **model_params)
        elif model_type == "SVM":
            model = SVC(random_state=42, probability=True, **model_params)
        else:
            model = RandomForestClassifier(random_state=42, **model_params)

        if w_train is not None:
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        except Exception:
            conf_matrix = []

        result = {
            "status": "success",
            "model_type": model_type,
            "model_params": model_params,
            "test_size": test_size,
            "dataset_size": len(_train) + len(_test),
            "test_samples": len(y_test),
            "performance": {
                "accuracy": round(acc, 4),
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "confusion_matrix": conf_matrix,
                "per_label_metrics": classification_report(y_test, y_pred, output_dict=True)
            },
            "fairness_analysis": {},
            "positive_class": str(positive_class_name)
        }

        if sensitive_columns:
            fairness_results = {}
            for sens_col in sensitive_columns:
                if sens_col not in X_test_raw.columns:
                    continue
                groups = X_test_raw[sens_col].fillna("Missing").astype(str)
                unique_groups = groups.unique()
                group_metrics = {}
                positive_rates = {}
                for group in unique_groups:
                    mask = (groups == group)
                    if mask.sum() == 0:
                        continue
                    y_test_g = y_test[mask.values]
                    y_pred_g = y_pred[mask.values]
                    g_acc = accuracy_score(y_test_g, y_pred_g)
                    g_f1 = f1_score(y_test_g, y_pred_g, average='macro', zero_division=0)
                    pred_pos_count = (y_pred_g == positive_label_idx).sum()
                    total_count = len(y_pred_g)
                    pos_rate = pred_pos_count / total_count if total_count > 0 else 0
                    actual_pos_count = (y_test_g == positive_label_idx).sum()
                    base_rate = actual_pos_count / total_count if total_count > 0 else 0
                    y_test_g_binary = (y_test_g == positive_label_idx).astype(int)
                    y_pred_g_binary = (y_pred_g == positive_label_idx).astype(int)
                    try:
                        tn, fp, fn, tp = confusion_matrix(
                            y_test_g_binary, y_pred_g_binary, labels=[0, 1]
                        ).ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    except Exception:
                        tn = fp = fn = tp = 0
                        tpr = tnr = fpr = fnr = 0
                    group_metrics[str(group)] = {
                        "accuracy": round(g_acc, 4),
                        "f1_macro": round(g_f1, 4),
                        "positive_rate": round(pos_rate, 4),
                        "base_rate": round(base_rate, 4),
                        "count": int(total_count),
                        "tpr": round(tpr, 4),
                        "tnr": round(tnr, 4),
                        "fpr": round(fpr, 4),
                        "fnr": round(fnr, 4),
                        "tp": int(tp),
                        "fp": int(fp),
                        "tn": int(tn),
                        "fn": int(fn)
                    }
                    positive_rates[str(group)] = pos_rate
                rates_list = list(positive_rates.values())
                if rates_list:
                    max_rate = max(rates_list)
                    min_rate = min(rates_list)
                    spd = max_rate - min_rate
                    non_zero_rates = [r for r in rates_list if r > 0]
                    if non_zero_rates and max_rate > 0:
                        min_non_zero_rate = min(non_zero_rates)
                        di = min_non_zero_rate / max_rate
                        min_rate_group = min(
                            (g for g, r in positive_rates.items() if r > 0),
                            key=lambda g: positive_rates[g]
                        )
                    else:
                        di = 0.0
                        min_rate_group = (
                            min(positive_rates, key=positive_rates.get) if positive_rates else "N/A"
                        )
                else:
                    spd = 0.0
                    di = 0.0
                    min_rate_group = "N/A"
                fairness_results[sens_col] = {
                    "groups": group_metrics,
                    "metrics": {
                        "statistical_parity_difference": round(spd, 4),
                        "disparate_impact": round(di, 4),
                        "max_positive_rate_group": (
                            max(positive_rates, key=positive_rates.get) if positive_rates else "N/A"
                        ),
                        "min_positive_rate_group": min_rate_group
                    }
                }
            result["fairness_analysis"] = fairness_results

        return result

    def train_and_evaluate_ml_model(self, dataset_name: str = None, target_column: str = None,
                                     sensitive_columns: list = None, test_size: float = 0.25,
                                     model_type: str = "Random Forest", model_params: dict = None,
                                     train_df=None, test_df=None) -> dict:
        try:
            if train_df is not None and test_df is not None:
                return self._eval_presplit(
                    train_df, test_df, target_column, sensitive_columns,
                    test_size, model_type, model_params or {}
                )
            path = self._resolve_path(dataset_name)
            
            na_values = ['?', 'NA', 'N/A', 'n/a', 'na', 'NULL', 'null', 'None', 'none', 
                        '', ' ', 'NaN', 'nan', '--', '..', 'missing', 'Missing', 
                        'unknown', 'Unknown', 'UNKNOWN', 'undefined', 'Undefined']
            
            df = pd.read_csv(path, na_values=na_values, keep_default_na=True)
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            df = df.dropna(subset=[target_column])
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            sample_weights_full = None
            if 'sample_weight' in X.columns:
                sample_weights_full = X['sample_weight']
                X = X.drop(columns=['sample_weight'])
            
            
            X_raw = X.copy()
            
            le_dict = {}
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = X[col].fillna("Missing").astype(str)
                X[col] = le.fit_transform(X[col])
                le_dict[col] = le
            
            for col in X.select_dtypes(include=['int64', 'float64']).columns:
                X[col] = X[col].fillna(X[col].median())
                
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            
            positive_label_idx = 1 if len(le_target.classes_) > 1 else 0
            positive_class_name = le_target.classes_[positive_label_idx]
            
            if sample_weights_full is not None:
                try:
                    X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
                        X, y, sample_weights_full, df.index, test_size=test_size, random_state=42, stratify=y
                    )
                except ValueError as e:
                    print(f"Warning: Stratified split failed ({str(e)}), using regular split")
                    X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
                        X, y, sample_weights_full, df.index, test_size=test_size, random_state=42
                    )
            else:
                w_train = None
                try:
                    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                        X, y, df.index, test_size=test_size, random_state=42, stratify=y
                    )
                except ValueError as e:
                    print(f"Warning: Stratified split failed ({str(e)}), using regular split")
                    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                        X, y, df.index, test_size=test_size, random_state=42
                    )
            
            X_test_raw = df.loc[idx_test].drop(columns=[target_column])
            model_params = model_params or {}
            
            if model_type == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000, **model_params)
            elif model_type == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=42, **model_params)
            elif model_type == "SVM":
                model = SVC(random_state=42, probability=True, **model_params)
            else: 
                model = RandomForestClassifier(random_state=42, **model_params)
            
            if w_train is not None:
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            except:
                conf_matrix = []
            
            result = {
                "status": "success",
                "model_type": model_type,
                "model_params": model_params,
                "test_size": test_size,
                "dataset_size": len(df),
                "test_samples": len(y_test),
                "performance": {
                    "accuracy": round(acc, 4),
                    "f1_macro": round(f1_macro, 4),
                    "f1_weighted": round(f1_weighted, 4),
                    "confusion_matrix": conf_matrix,
                    "per_label_metrics": classification_report(y_test, y_pred, output_dict=True)
                },
                "fairness_analysis": {},
                "positive_class": str(positive_class_name)
            }
            
            if sensitive_columns:
                fairness_results = {}
                
                for sens_col in sensitive_columns:
                    if sens_col not in X_test_raw.columns:
                        continue
                        
                    groups = X_test_raw[sens_col].fillna("Missing").astype(str)
                    unique_groups = groups.unique()
                    
                    group_metrics = {}
                    positive_rates = {}
                    
                    for group in unique_groups:
                        mask = (groups == group)
                        if mask.sum() == 0:
                            continue
                            
                        y_test_g = y_test[mask]
                        y_pred_g = y_pred[mask]
                        
                        g_acc = accuracy_score(y_test_g, y_pred_g)
                        g_f1 = f1_score(y_test_g, y_pred_g, average='macro', zero_division=0)
                        
                        pred_pos_count = (y_pred_g == positive_label_idx).sum()
                        total_count = len(y_pred_g)
                        pos_rate = pred_pos_count / total_count if total_count > 0 else 0
                        
                        actual_pos_count = (y_test_g == positive_label_idx).sum()
                        base_rate = actual_pos_count / total_count if total_count > 0 else 0
                        
                        y_test_g_binary = (y_test_g == positive_label_idx).astype(int)
                        y_pred_g_binary = (y_pred_g == positive_label_idx).astype(int)
                        
                        try:
                            tn, fp, fn, tp = confusion_matrix(y_test_g_binary, y_pred_g_binary, labels=[0, 1]).ravel()
                            
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  
                            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 
                            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  
                        except Exception:
                            tn = fp = fn = tp = 0
                            tpr = tnr = fpr = fnr = 0

                        group_metrics[str(group)] = {
                            "accuracy": round(g_acc, 4),
                            "f1_macro": round(g_f1, 4),
                            "positive_rate": round(pos_rate, 4),
                            "base_rate": round(base_rate, 4),
                            "count": int(total_count),
                            "tpr": round(tpr, 4),
                            "tnr": round(tnr, 4),
                            "fpr": round(fpr, 4),
                            "fnr": round(fnr, 4),
                            "tp": int(tp),
                            "fp": int(fp), 
                            "tn": int(tn),
                            "fn": int(fn)
                        }
                        positive_rates[str(group)] = pos_rate
                    
                    rates_list = list(positive_rates.values())
                    if rates_list:
                        max_rate = max(rates_list)
                        min_rate = min(rates_list)
                        
                        spd = max_rate - min_rate
                        non_zero_rates = [r for r in rates_list if r > 0]
                        if non_zero_rates and max_rate > 0:
                            min_non_zero_rate = min(non_zero_rates)
                            di = min_non_zero_rate / max_rate
                            # Use the group with the non-zero minimum for consistency with DI
                            min_rate_group = min(
                                (g for g, r in positive_rates.items() if r > 0),
                                key=lambda g: positive_rates[g]
                            )
                        else:
                            di = 0.0
                            min_rate_group = min(positive_rates, key=positive_rates.get) if positive_rates else "N/A"
                    else:
                        spd = 0.0
                        di = 0.0
                        min_rate_group = "N/A"
                    
                    fairness_results[sens_col] = {
                        "groups": group_metrics,
                        "metrics": {
                            "statistical_parity_difference": round(spd, 4),
                            "disparate_impact": round(di, 4),
                            "max_positive_rate_group": max(positive_rates, key=positive_rates.get) if positive_rates else "N/A",
                            "min_positive_rate_group": min_rate_group
                        }
                    }
                
                result["fairness_analysis"] = fairness_results
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": f"ML model error: {str(e)}"}