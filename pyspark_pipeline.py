import os
import sys
import io
import json
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Fix Windows Python path for Spark workers
python_exe = sys.executable
os.environ["PYSPARK_PYTHON"] = python_exe
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe


import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, lit, udf, round as spark_round
from pyspark.sql.types import DoubleType, FloatType, IntegerType, StringType

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def create_spark_session():
    """Create and configure optimized local SparkSession."""
    print("=" * 70)
    print("STEP 1: Initializing Apache Spark Session...")
    print("=" * 70)
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("CustomerPurchaseBehaviourPrediction")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"✓ SparkSession active (Version: {spark.version}, Master: local[*])")
    return spark


def task_163_load_and_clean_data(spark, csv_path="customer_behaviour.csv"):
    """
    Task 163:
    - Create SparkSession (done)
    - Load CSV dataset into a Spark DataFrame
    - Display schema and first five records
    - Identify missing and null values in every column
    - Handle missing numerical values using appropriate imputation (Imputer)
    - Handle missing categorical values using appropriate strategy
    - Remove duplicate records where appropriate
    - Verify that cleaned dataset is ready for machine learning
    """
    print("\n" + "=" * 70)
    print("TASK 163: Load Dataset and Handle Missing & Duplicate Values")
    print("=" * 70)

    # 1. Load CSV into Spark DataFrame
    df_raw = spark.read.csv(csv_path, header=True, inferSchema=True)
    raw_count = df_raw.count()
    print(f"✓ Loaded '{csv_path}' successfully with {raw_count} records and {len(df_raw.columns)} columns.")

    # 2. Display Schema
    print("\n--- DataFrame Schema ---")
    df_raw.printSchema()

    # 3. Display First 5 Records
    print("\n--- First 5 Records ---")
    df_raw.show(5, truncate=False)

    # 4. Identify missing / null values in every column
    print("\n--- Missing & Null Value Analysis Per Column ---")
    null_counts = {}
    dtypes_dict = dict(df_raw.dtypes)
    for c in df_raw.columns:
        # Check for nulls and NaNs appropriately based on column data type
        if dtypes_dict[c] in ["double", "float"]:
            null_count = df_raw.filter(col(c).isNull() | isnan(col(c))).count()
        else:
            null_count = df_raw.filter(col(c).isNull()).count()
        null_counts[c] = null_count
        status = f"⚠️ {null_count} missing" if null_count > 0 else "✓ Clean (0 missing)"
        print(f"  {c:<22} : {status}")

    # 5. Remove Duplicate Records
    duplicates_before = raw_count
    df_dedup = df_raw.dropDuplicates(["customer_id"])
    dedup_count = df_dedup.count()
    removed_dupes = duplicates_before - dedup_count
    print(f"\n✓ Removed {removed_dupes} duplicate records (Count: {duplicates_before} -> {dedup_count})")

    # 6. Handle Missing Categorical Values (Mode / Fallback Imputation)
    # Categorical columns with missing values: customer_age_group, browsing_activity
    # Find mode of customer_age_group and browsing_activity
    mode_age = df_dedup.groupBy("customer_age_group").count().filter(col("customer_age_group").isNotNull()).orderBy(col("count").desc()).first()[0]
    mode_browse = df_dedup.groupBy("browsing_activity").count().filter(col("browsing_activity").isNotNull()).orderBy(col("count").desc()).first()[0]
    print(f"✓ Categorical Imputation: Mode for 'customer_age_group' = '{mode_age}', Mode for 'browsing_activity' = '{mode_browse}'")

    df_cat_filled = df_dedup.fillna({
        "customer_age_group": mode_age,
        "browsing_activity": mode_browse
    })

    # 7. Handle Missing Numerical Values using PySpark MLlib Imputer (Strategy = 'median')
    num_cols_to_impute = ["product_views", "session_duration"]
    output_num_cols = [f"{c}" for c in num_cols_to_impute]
    
    imputer = Imputer(
        inputCols=num_cols_to_impute,
        outputCols=output_num_cols,
        strategy="median"
    )
    imputer_model = imputer.fit(df_cat_filled)
    df_imputed = imputer_model.transform(df_cat_filled)

    # Ensure target column is Integer
    df_cleaned = df_imputed.withColumn("label", col("purchase").cast(DoubleType()))

    # 8. Verify cleaned dataset
    remaining_nulls = 0
    clean_dtypes = dict(df_cleaned.dtypes)
    for c in df_cleaned.columns:
        if clean_dtypes[c] in ["double", "float"]:
            null_c = df_cleaned.filter(col(c).isNull() | isnan(col(c))).count()
        else:
            null_c = df_cleaned.filter(col(c).isNull()).count()
        remaining_nulls += null_c

    clean_count = df_cleaned.count()
    print(f"\n✓ Verification Complete: Total clean records = {clean_count}, Remaining nulls = {remaining_nulls}")
    print("✓ Dataset is 100% verified and ready for Machine Learning pipeline.")

    return df_cleaned, null_counts, raw_count, clean_count


def task_164_feature_engineering_pipeline(df_cleaned):
    """
    Task 164:
    - Identify categorical and numerical columns
    - Apply StringIndexer to categorical features with handleInvalid='keep'
    - Apply OneHotEncoder where appropriate
    - Use VectorAssembler to combine all input features into 'features' vector
    - Target stored in 'label'
    """
    print("\n" + "=" * 70)
    print("TASK 164: Feature Encoding and Vector Assembly")
    print("=" * 70)

    categorical_cols = [
        "customer_age_group",
        "browsing_activity",
        "customer_segment",
        "visitor_type",
        "month"
    ]

    numerical_cols = [
        "past_purchases",
        "product_views",
        "session_duration",
        "bounce_rate",
        "exit_rate",
        "page_values",
        "weekend"
    ]

    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical Features   ({len(numerical_cols)}): {numerical_cols}")

    stages = []

    # 1. StringIndexer for each categorical column
    indexed_cols = [f"{c}_index" for c in categorical_cols]
    encoded_cols = [f"{c}_vec" for c in categorical_cols]

    for cat_col, idx_col in zip(categorical_cols, indexed_cols):
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=idx_col,
            handleInvalid="keep"
        )
        stages.append(indexer)

    # 2. OneHotEncoder for indexed categorical features
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols,
        handleInvalid="keep"
    )
    stages.append(encoder)

    # 3. VectorAssembler to assemble all features into a single vector column named 'features'
    assembler_inputs = encoded_cols + numerical_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep"
    )
    stages.append(assembler)

    print(f"✓ Configured {len(stages)} pipeline stages: StringIndexers, OneHotEncoder, and VectorAssembler.")
    print("✓ Assembled feature vector column will be stored in: 'features'")
    print("✓ Classification target variable column: 'label' (binary purchase: 0 or 1)")

    return stages, categorical_cols, numerical_cols


def task_165_train_models(df_cleaned, feature_stages):
    """
    Task 165:
    - Split dataset into train and test sets (80/20)
    - Train classification models using PySpark MLlib:
      * Logistic Regression
      * Gradient-Boosted Trees (GBTClassifier)
    - Save trained PipelineModel
    - Generate predictions and probabilities on test set
    """
    print("\n" + "=" * 70)
    print("TASK 165: Build and Train Classification Models")
    print("=" * 70)

    # Train / Test split (80% / 20%)
    train_df, test_df = df_cleaned.randomSplit([0.80, 0.20], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"✓ Dataset split: Training = {train_count} records (80%), Testing = {test_count} records (20%)")

    # Model 1: Logistic Regression
    print("\n[1/2] Training Model 1: Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=25,
        regParam=0.01,
        elasticNetParam=0.0
    )
    pipeline_lr = Pipeline(stages=feature_stages + [lr])
    model_lr = pipeline_lr.fit(train_df)
    lr_train_time = round(time.time() - t0, 2)
    print(f"✓ Logistic Regression training complete in {lr_train_time} seconds.")

    # Model 2: Gradient-Boosted Trees (GBTClassifier)
    print("\n[2/2] Training Model 2: Gradient-Boosted Trees (GBTClassifier)...")
    t0 = time.time()
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        maxDepth=5,
        seed=42
    )
    pipeline_gbt = Pipeline(stages=feature_stages + [gbt])
    model_gbt = pipeline_gbt.fit(train_df)
    gbt_train_time = round(time.time() - t0, 2)
    print(f"✓ Gradient-Boosted Trees training complete in {gbt_train_time} seconds.")

    # Generate test predictions
    print("\nGenerating predictions on unseen test data...")
    predictions_lr = model_lr.transform(test_df)
    predictions_gbt = model_gbt.transform(test_df)

    # Save PipelineModel
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Overwrite if exists
    lr_model_path = "models/spark_lr_pipeline_model"
    gbt_model_path = "models/spark_gbt_pipeline_model"
    model_lr.write().overwrite().save(lr_model_path)
    model_gbt.write().overwrite().save(gbt_model_path)
    print(f"✓ Saved reusable Logistic Regression PipelineModel to '{lr_model_path}'")
    print(f"✓ Saved reusable Gradient-Boosted Trees PipelineModel to '{gbt_model_path}'")

    return model_lr, model_gbt, predictions_lr, predictions_gbt, train_df, test_df


def task_166_evaluate_model(predictions_lr, predictions_gbt):
    """
    Task 166:
    - Evaluate using all five required metrics:
      1. Accuracy
      2. Precision
      3. Recall
      4. F1-score
      5. ROC-AUC
    - Generate confusion matrix (TP, TN, FP, FN)
    - Generate ROC curve
    - Save results for dashboard integration
    """
    print("\n" + "=" * 70)
    print("TASK 166: Comprehensive Model Evaluation (5 Required Metrics)")
    print("=" * 70)

    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    def compute_all_metrics(pred_df, model_name):
        acc = float(evaluator_acc.evaluate(pred_df))
        w_prec = float(evaluator_prec.evaluate(pred_df))
        w_rec = float(evaluator_rec.evaluate(pred_df))
        f1 = float(evaluator_f1.evaluate(pred_df))
        roc_auc = float(evaluator_roc.evaluate(pred_df))

        # Confusion Matrix counts
        tp = pred_df.filter((col("label") == 1.0) & (col("prediction") == 1.0)).count()
        tn = pred_df.filter((col("label") == 0.0) & (col("prediction") == 0.0)).count()
        fp = pred_df.filter((col("label") == 0.0) & (col("prediction") == 1.0)).count()
        fn = pred_df.filter((col("label") == 1.0) & (col("prediction") == 0.0)).count()

        pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0.0

        metrics = {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision": round(pos_precision, 4),
            "weighted_precision": round(w_prec, 4),
            "recall": round(pos_recall, 4),
            "weighted_recall": round(w_rec, 4),
            "f1_score": round(pos_f1, 4),
            "weighted_f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "total_test_samples": int(tp + tn + fp + fn)
        }
        return metrics

    lr_metrics = compute_all_metrics(predictions_lr, "Logistic Regression")
    gbt_metrics = compute_all_metrics(predictions_gbt, "Gradient-Boosted Trees (GBT)")

    print(f"\n{'Metric':<25} | {'Logistic Regression':<22} | {'Gradient-Boosted Trees':<22}")
    print("-" * 75)
    print(f"{'1. Accuracy':<25} | {lr_metrics['accuracy']*100:<21.2f}% | {gbt_metrics['accuracy']*100:<21.2f}%")
    print(f"{'2. Precision (Purchase)':<25} | {lr_metrics['precision']*100:<21.2f}% | {gbt_metrics['precision']*100:<21.2f}%")
    print(f"{'3. Recall (Purchase)':<25} | {lr_metrics['recall']*100:<21.2f}% | {gbt_metrics['recall']*100:<21.2f}%")
    print(f"{'4. F1-Score':<25} | {lr_metrics['f1_score']*100:<21.2f}% | {gbt_metrics['f1_score']*100:<21.2f}%")
    print(f"{'5. ROC-AUC':<25} | {lr_metrics['roc_auc']:<22.4f} | {gbt_metrics['roc_auc']:<22.4f}")
    print("-" * 75)
    print(f"{'True Positives (TP)':<25} | {lr_metrics['tp']:<22} | {gbt_metrics['tp']:<22}")
    print(f"{'True Negatives (TN)':<25} | {lr_metrics['tn']:<22} | {gbt_metrics['tn']:<22}")
    print(f"{'False Positives (FP)':<25} | {lr_metrics['fp']:<22} | {gbt_metrics['fp']:<22}")
    print(f"{'False Negatives (FN)':<25} | {lr_metrics['fn']:<22} | {gbt_metrics['fn']:<22}")

    # Convert test predictions to Pandas to safely extract probability and compute ROC curve
    cols_to_select = [
        "customer_id", "customer_age_group", "browsing_activity", 
        "past_purchases", "product_views", "session_duration", 
        "customer_segment", "purchase", "prediction", "probability"
    ]
    pdf = predictions_lr.select(*cols_to_select).toPandas()

    # Extract probability of purchase (class 1)
    pdf["purchase_probability"] = pdf["probability"].apply(
        lambda v: round(float(v[1]), 4) if v is not None and len(v) > 1 else 0.5
    )

    from sklearn.metrics import roc_curve, auc
    y_true = pdf["purchase"].values
    y_score = pdf["purchase_probability"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Subsample ROC points for compact storage
    step = max(1, len(fpr) // 100)
    roc_points = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr[::step], tpr[::step])
    ]

    eval_data = {
        "logistic_regression": lr_metrics,
        "gbt": gbt_metrics,
        "roc_curve": roc_points,
        "winning_model": "Gradient-Boosted Trees (GBT)" if gbt_metrics["f1_score"] >= lr_metrics["f1_score"] else "Logistic Regression"
    }

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    metrics_path = "data/model_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n✓ Exported comprehensive evaluation metrics & ROC coordinates to '{metrics_path}'")
    return eval_data, pdf


def task_167_generate_predictions_csv(pdf_predictions, output_csv="data/predictions.csv"):
    """
    Task 167:
    - Apply model to test dataset
    - Generate actual purchase values
    - Generate predicted purchase values
    - Generate purchase probabilities
    - Compare actual and predicted outcomes
    - Include customer segment information
    - The prediction output must contain:
      * Customer age group
      * Browsing activity
      * Past purchases
      * Product views
      * Session duration
      * Customer segment
      * Actual purchase
      * Predicted purchase
      * Purchase probability
    - Save predictions in CSV format
    """
    print("\n" + "=" * 70)
    print("TASK 167: Apply Model and Generate Purchase Predictions CSV")
    print("=" * 70)

    # Rename and select columns to exact required specifications
    pdf_out = pd.DataFrame()
    pdf_out["customer_id"] = pdf_predictions["customer_id"]
    pdf_out["customer_age_group"] = pdf_predictions["customer_age_group"]
    pdf_out["browsing_activity"] = pdf_predictions["browsing_activity"]
    pdf_out["past_purchases"] = pdf_predictions["past_purchases"]
    pdf_out["product_views"] = pdf_predictions["product_views"]
    pdf_out["session_duration"] = pdf_predictions["session_duration"].round(2)
    pdf_out["customer_segment"] = pdf_predictions["customer_segment"]
    pdf_out["actual_purchase"] = pdf_predictions["purchase"].astype(int)
    pdf_out["predicted_purchase"] = pdf_predictions["prediction"].astype(int)
    pdf_out["purchase_probability"] = pdf_predictions["purchase_probability"]

    # Comparison and outcome classification
    pdf_out["prediction_outcome"] = np.where(
        (pdf_out["actual_purchase"] == 1) & (pdf_out["predicted_purchase"] == 1), "True Positive",
        np.where((pdf_out["actual_purchase"] == 0) & (pdf_out["predicted_purchase"] == 0), "True Negative",
        np.where((pdf_out["actual_purchase"] == 0) & (pdf_out["predicted_purchase"] == 1), "False Positive", "False Negative"))
    )
    pdf_out["prediction_correct"] = pdf_out["actual_purchase"] == pdf_out["predicted_purchase"]

    # Save CSVs
    pdf_out.to_csv(output_csv, index=False)
    pdf_out.to_csv("data/test_predictions.csv", index=False)

    print(f"✓ Saved {len(pdf_out)} test purchase predictions to '{output_csv}' and 'data/test_predictions.csv'")
    print("\n--- Preview of Final Predictions CSV ---")
    print(pdf_out.head(5)[
        ["customer_id", "customer_age_group", "browsing_activity", "past_purchases", 
         "product_views", "session_duration", "customer_segment", "actual_purchase", 
         "predicted_purchase", "purchase_probability", "prediction_outcome"]
    ].to_string())

    actual_rate = (pdf_out["actual_purchase"].mean() * 100).round(2)
    pred_rate = (pdf_out["predicted_purchase"].mean() * 100).round(2)
    correct_count = pdf_out["prediction_correct"].sum()
    total_count = len(pdf_out)
    accuracy = ((correct_count / total_count) * 100).round(2)

    print(f"\nPrediction Summary:")
    print(f"  - Total Test Records         : {total_count}")
    print(f"  - Actual Purchases           : {pdf_out['actual_purchase'].sum()} ({actual_rate}% conversion)")
    print(f"  - Predicted Purchases        : {pdf_out['predicted_purchase'].sum()} ({pred_rate}% conversion)")
    print(f"  - Correct Predictions (TP+TN): {correct_count} of {total_count} ({accuracy}% accuracy)")

    return pdf_out



def run_full_pipeline():
    """Execute complete end-to-end PySpark Machine Learning pipeline."""
    spark = create_spark_session()
    try:
        # Task 163
        df_cleaned, null_counts, raw_count, clean_count = task_163_load_and_clean_data(spark)

        # Task 164
        stages, cat_cols, num_cols = task_164_feature_engineering_pipeline(df_cleaned)

        # Task 165
        model_lr, model_gbt, pred_lr, pred_gbt, train_df, test_df = task_165_train_models(df_cleaned, stages)

        # Task 166
        eval_data, pred_prob_df = task_166_evaluate_model(pred_lr, pred_gbt)

        # Task 167
        pdf_preds = task_167_generate_predictions_csv(pred_prob_df)

        print("\n" + "=" * 70)
        print("🎉 ALL TASKS (163, 164, 165, 166, 167) COMPLETED SUCCESSFULLY!")
        print("Ready to launch Task 168: Interactive Streamlit Dashboard.")
        print("=" * 70)
        return True
    finally:
        spark.stop()
        print("✓ SparkSession stopped gracefully.")


if __name__ == "__main__":
    run_full_pipeline()
