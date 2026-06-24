import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report
import traceback

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="AI Feature Selection & Elimination",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for clean and professional UI
st.markdown("""
<style>
    .step-header { color: #2563eb; font-weight: 700; font-size: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
    .stat-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1.5rem; text-align: center; }
    .stat-value { font-size: 2rem; font-weight: 700; color: #2563eb; }
    .stat-label { font-size: 0.875rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Feature Elimination & Selection (RFECV)")
st.markdown("Automated tool to find important variables and eliminate useless ones using Recursive Feature Elimination with Cross-Validation (RFECV).")

# ==========================================
# STEP 1: UPLOAD DATASET
# ==========================================
st.markdown('<div class="step-header">Step 1: Upload Dataset</div>', unsafe_allow_html=True)
st.markdown("""
**CSV File Requirements:**
* **Column 1:** Data ID (e.g., File Name, Student ID)
* **Column 2:** Answer or Label (e.g., Male/Female, Normal/Risk)
* **Column 3 onwards:** Numeric values of the variables (Features) to be analyzed
""")

uploaded_file = st.file_uploader("Drag and drop your CSV file here", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.markdown("##### 📄 Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Check structure and count
        id_col = df.columns[0]
        label_col = df.columns[1]
        feature_cols = df.columns[2:]
        
        n_samples = len(df)
        n_features = len(feature_cols)
        
        # Display data statistics
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.markdown(f'<div class="stat-card"><div class="stat-label">Number of Samples</div><div class="stat-value">{n_samples}</div></div>', unsafe_allow_html=True)
        col_s2.markdown(f'<div class="stat-card"><div class="stat-label">Number of Features</div><div class="stat-value">{n_features}</div></div>', unsafe_allow_html=True)
        col_s3.markdown(f'<div class="stat-card"><div class="stat-label">Number of Target Classes</div><div class="stat-value">{df[label_col].nunique()}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ⚠️ Evaluation Condition 1: Samples must be greater than Features
        if n_samples <= n_features:
            st.error(f"🚨 **Statistical Error (Dimensionality Curse):** The number of data rows ({n_samples} rows) is less than or equal to the number of variables ({n_features} variables).")
            st.warning("👉 **Recommendation:** You must increase the number of samples or manually remove some variables first so the system can work accurately without overfitting.")
            st.stop()
            
        # ⚠️ Evaluation Condition 2: Must have at least 2 classes to perform ML
        elif df[label_col].nunique() < 2:
            st.error(f"🚨 **Data Error:** Your dataset contains only 1 class/label in the '{label_col}' column. Machine learning models require at least 2 different classes (e.g., Risk vs Control) to learn patterns and make classifications.")
            st.stop()
            
        else:
            st.success("✅ Data structure meets all criteria. Ready for analysis.")
            
            # ==========================================
            # STEP 2: PROCESS & VISUALIZE
            # ==========================================
            st.markdown('<div class="step-header">Step 2: Feature Analysis (RFECV)</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Analysis & Simulations", type="primary"):
                with st.spinner("Training AI models, selecting features, and running simulations... (This may take a moment)"):
                    
                    # 1. Prepare X and y data
                    X = df[feature_cols]
                    y_raw = df[label_col]
                    
                    # Convert text labels to numbers
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)
                    
                    # Check minimum class count
                    min_class_count = pd.Series(y).value_counts().min()
                    cv_folds = min(5, min_class_count) 
                    
                    if cv_folds < 2:
                        st.error(f"🚨 **Imbalance Error:** One of your classes has only {min_class_count} sample. RFECV requires at least 2 samples per class to perform cross-validation.")
                        st.stop()
                    
                    # 2. Setup model and RFECV
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    
                    rfecv = RFECV(
                        estimator=rf_model,
                        step=1, 
                        cv=StratifiedKFold(cv_folds),
                        scoring='accuracy',
                        min_features_to_select=1
                    )
                    
                    # 3. Start training
                    rfecv.fit(X, y)
                    
                    # 4. Extract results
                    optimal_num_features = rfecv.n_features_
                    selected_features_mask = rfecv.support_
                    
                    selected_features = np.array(feature_cols)[selected_features_mask]
                    eliminated_features = np.array(feature_cols)[~selected_features_mask]
                    
                    # Train again with selected features to extract Importance
                    rf_model.fit(X[selected_features], y)
                    importances = rf_model.feature_importances_
                    indices = np.argsort(importances)[::-1] 
                    
                    st.success(f"🎉 Selection Complete! The AI recommends keeping **{optimal_num_features}** features out of the original {n_features}.")
                    
                    # --- Step 2 Visualizations ---
                    st.markdown("### 📊 RFECV Academic Summary")
                    
                    plt.style.use('default')
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                    
                    try:
                        cv_results = rfecv.cv_results_['mean_test_score']
                    except KeyError:
                        cv_results = rfecv.grid_scores_
                        
                    ax1.plot(range(1, len(cv_results) + 1), cv_results, marker='o', color='#2563eb', linewidth=2, markersize=6)
                    ax1.set_title('Optimal Number of Features vs. Cross-Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
                    ax1.set_xlabel('Number of Features Selected', fontsize=12)
                    ax1.set_ylabel('Cross-Validation Accuracy', fontsize=12)
                    ax1.axvline(x=optimal_num_features, color='red', linestyle='--', label=f'Optimal: {optimal_num_features} Features')
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    ax1.legend()
                    
                    sorted_features = [selected_features[i] for i in indices]
                    sorted_importances = importances[indices]
                    
                    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(sorted_importances)))
                    
                    bars = ax2.bar(range(len(sorted_importances)), sorted_importances, color=colors)
                    ax2.set_title(f'Feature Importances (Top {optimal_num_features} Selected Features)', fontsize=14, fontweight='bold', pad=15)
                    ax2.set_xlabel('Features', fontsize=12)
                    ax2.set_ylabel('Importance Coefficient', fontsize=12)
                    ax2.set_xticks(range(len(sorted_importances)))
                    ax2.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=10)
                    ax2.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    plt.tight_layout(pad=3.0)
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    # ==========================================
                    # DATA TABLES (Restored)
                    # ==========================================
                    col_t1, col_t2 = st.columns(2)
                    
                    with col_t1:
                        st.markdown("#### ✅ Selected Features")
                        df_selected = pd.DataFrame({
                            "Feature Name": sorted_features,
                            "Importance Score": [f"{score:.4f}" for score in sorted_importances]
                        })
                        st.dataframe(df_selected, use_container_width=True, hide_index=True)
                        
                    with col_t2:
                        st.markdown("#### ❌ Eliminated Features")
                        if len(eliminated_features) > 0:
                            df_eliminated = pd.DataFrame({"Eliminated Feature Name": eliminated_features})
                            st.dataframe(df_eliminated, use_container_width=True, hide_index=True)
                        else:
                            st.success("No features were eliminated.")

                    st.markdown("---")
                    
                    # ==========================================
                    # DATA PREPARATION FOR SIMULATIONS
                    # ==========================================
                    # Filter dataset to only keep the selected/optimized features
                    X_opt = X[selected_features]
                    
                    # Split into train and test sets (70% Train, 30% Test)
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_opt, y, test_size=0.3, random_state=42, stratify=y if min_class_count >= 2 else None
                        )
                    except ValueError:
                        # Fallback if dataset is too small or imbalanced to stratify
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_opt, y, test_size=0.3, random_state=42
                        )
                    
                    # ==========================================
                    # STEP 3: RANDOM FOREST SIMULATION
                    # ==========================================
                    st.markdown('<div class="step-header mt-8">Step 3: Random Forest Simulation (Optimized Dataset)</div>', unsafe_allow_html=True)
                    st.markdown("Testing a **Random Forest Classifier** using *only* the recommended features. The decision tree visual below illustrates how the model derives its rules.")
                    
                    # Train RF Simulation Model
                    rf_sim = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight='balanced')
                    rf_sim.fit(X_train, y_train)
                    rf_pred = rf_sim.predict(X_test)
                    rf_acc = accuracy_score(y_test, rf_pred)
                    
                    st.info(f"🎯 **Random Forest Test Accuracy:** {rf_acc * 100:.2f}%")
                    
                    # Visualize a single decision tree from the forest
                    st.markdown("#### 🌳 Sample Decision Tree Logic")
                    fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
                    
                    class_names_str = [str(c) for c in le.classes_]
                    plot_tree(
                        rf_sim.estimators_[0], 
                        feature_names=list(selected_features), 
                        class_names=class_names_str, 
                        filled=True, 
                        rounded=True, 
                        fontsize=9, 
                        max_depth=3,
                        ax=ax_tree
                    )
                    plt.title("Visualized Rules of Estimator Tree #0", pad=20, fontsize=14, fontweight='bold')
                    st.pyplot(fig_tree)
                    
                    st.markdown("---")
                    
                    # ==========================================
                    # STEP 4: SVM SIMULATION
                    # ==========================================
                    st.markdown('<div class="step-header mt-8">Step 4: Support Vector Machine (SVM) Simulation</div>', unsafe_allow_html=True)
                    st.markdown("Testing a **Linear Support Vector Machine (SVM)** to observe how features push the decision boundary linearly.")
                    
                    # Build SVM Pipeline (Must include standard scaling)
                    svm_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svm', SVC(kernel='linear', class_weight='balanced', random_state=42))
                    ])
                    
                    # Train SVM Simulation Model
                    svm_pipeline.fit(X_train, y_train)
                    svm_pred = svm_pipeline.predict(X_test)
                    svm_acc = accuracy_score(y_test, svm_pred)
                    
                    st.info(f"🎯 **SVM Test Accuracy:** {svm_acc * 100:.2f}%")
                    
                    # Extract and visualize SVM Coefficients (Feature Weights)
                    st.markdown("#### ⚖️ SVM Feature Weights (Linear Coefficients)")
                    st.caption("Positive weights push the prediction towards Class 1. Negative weights push towards Class 0.")
                    
                    # Handle binary vs multiclass coefs
                    if len(le.classes_) == 2:
                        coefs = svm_pipeline.named_steps['svm'].coef_[0]
                    else:
                        # For multiclass, take the mean absolute importance across all OvR classifiers
                        coefs = np.mean(np.abs(svm_pipeline.named_steps['svm'].coef_), axis=0)
                        
                    # Sort coefficients by absolute magnitude for better visualization
                    sorted_idx = np.argsort(np.abs(coefs))[::-1]
                    sorted_coef_features = [selected_features[i] for i in sorted_idx]
                    sorted_coefs = [coefs[i] for i in sorted_idx]
                    
                    fig_svm, ax_svm = plt.subplots(figsize=(10, 6))
                    bar_colors = ['#10b981' if c > 0 else '#ef4444' for c in sorted_coefs]
                    
                    ax_svm.bar(range(len(sorted_coefs)), sorted_coefs, color=bar_colors)
                    ax_svm.axhline(0, color='black', linewidth=1)
                    ax_svm.set_title("SVM Feature Importance Weights", fontsize=14, fontweight='bold', pad=15)
                    ax_svm.set_ylabel('Coefficient Magnitude (Weight)', fontsize=12)
                    ax_svm.set_xticks(range(len(sorted_coefs)))
                    ax_svm.set_xticklabels(sorted_coef_features, rotation=45, ha='right', fontsize=10)
                    ax_svm.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig_svm)
                    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.expander("Show detailed error log").code(traceback.format_exc())