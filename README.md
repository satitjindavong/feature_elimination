# **🔍 AI Feature Selection & Elimination Tool**

## **1\. Program Objective**

This program is a generic web application built to **"filter and discover the most important features"** in your dataset.

In Machine Learning, if we input garbage data or useless features, the model will become confused and over-memorize the data (Overfitting). This program acts as an AI assistant to **analyze feature importance** and **automatically eliminate useless features**, making your model highly accurate, faster, and providing results that can be reliably explained in an academic or professional setting.

## **2\. Installation & Usage**

### **Prerequisites**

Ensure that your machine has **Python 3.8** or newer installed.

### **Installation**

Open your Terminal or Command Prompt, navigate to the project folder, and run the following command to install the required libraries from the requirements.txt file:

pip install \-r requirements.txt

*(Main libraries used: streamlit, pandas, numpy, matplotlib, scikit-learn)*

### **Running the App**

Type the following command to run the application:

streamlit run feature\_elimination.py

The program will automatically open a new tab in your web browser and is ready to use immediately.

## **3\. How to Use**

1. **Prepare your CSV file:** Prepare your data as a .csv file with the following structure:  
   * **Column 1:** Reference code or ID (e.g., File Name, Patient ID)  
   * **Column 2:** Target variable or Label (e.g., Normal/Risk, Male/Female)  
   * **Column 3 onwards:** Numeric values of the variables (Features)  
2. **Upload Data (Step 1):** Drag and drop the CSV file into the upload box. The program will verify if the data structure is statistically valid (the number of rows must be greater than the number of features).  
3. **Run Analysis (Step 2):** Click the **"🚀 Run Analysis & Simulations"** button.  
4. **View Results:** Grab a cup of coffee and wait a moment. The program will automatically generate academic graphs, tables of selected/eliminated features, and simulate the AI's decision-making process.

## **4\. Core AI Concepts (Simplified)**

This program is powered by 3 main Machine Learning algorithms:

### **🧠 1\. RFECV (Recursive Feature Elimination with Cross-Validation)**

* **Role:** The "HR Manager" who eliminates bad candidates.  
* **Concept:** It creates a test model to see which feature performs the worst. It then **eliminates the worst feature one by one (Recursive)** and re-tests the accuracy repeatedly (Cross-Validation) until it discovers exactly how many features should be kept to achieve the highest possible model accuracy.

### **🌳 2\. Random Forest Classifier**

* **Role:** The "Judging Panel" (Used to find feature importance and simulate the model).  
* **Concept:** It builds multiple "Decision Trees" (like a forest). Each tree randomly looks at the data and sets its own rules (e.g., if Pitch \> 30, go left). Finally, all trees vote on the final answer. This model is excellent at revealing **which features were used most often to create rules (Feature Importance)**.

### **📏 3\. SVM (Support Vector Machine \- Linear Kernel)**

* **Role:** The "Boundary Drawer".  
* **Concept:** Imagine data points scattered on a piece of paper. SVM tries to find **"a ruler to draw a straight line"** that separates the 2 groups of data with the widest possible gap (Margin). Using a Linear SVM helps us understand **how much weight and in which direction each feature pushes the data toward a specific class**.

## **5\. Understanding the Output**

After the program finishes processing, you will find 4 main outputs that can be directly included in your report:

1. **Accuracy Curve Line Graph:** \* **X-axis:** Number of features used / **Y-axis:** Accuracy  
   * **Meaning:** The higher the point on the graph, the higher the accuracy. The red dashed line indicates the **"most optimal and worthwhile"** number of features to keep.  
2. **Feature Importances Bar Chart:**  
   * **Meaning:** Displays the Gini Importance values. The feature with the tallest bar is the **"hero"** that helps the AI classify the data best.  
3. **Sample Decision Tree Logic Diagram:**  
   * **Meaning:** Reveals the "brain" of the AI by showing which numerical conditions it uses to make decisions (e.g., F0 Mean \<= 120 go left / \> 120 go right). This helps non-technical people understand the AI's reasoning (Explainable AI).  
4. **SVM Feature Weights Bar Chart:**  
   * **Meaning:** Shows the "weight and direction" of each feature.  
   * **Green bar (Positive value):** This feature pushes the AI to predict "Class 1".  
   * **Red bar (Negative value):** This feature pushes the AI to predict "Class 0".