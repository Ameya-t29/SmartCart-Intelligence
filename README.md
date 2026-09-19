# SmartCart Intelligence — Flask Unsupervised ML Website

This project converts the `tut_smartcart.ipynb` customer-segmentation workflow into an interactive Flask dashboard.

## Dataset

Put the original dataset here:

`data/smartcart_customers.csv`

The notebook expects columns such as:

- ID
- Year_Birth
- Education
- Marital_Status
- Income
- Kidhome
- Teenhome
- Dt_Customer
- Recency
- MntWines
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases
- NumWebVisitsMonth
- Response
- Complain

## Run

### Windows PowerShell / VS Code terminal

```powershell
cd "YOUR\SmartCart_Web"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open:

`http://127.0.0.1:5000`

If PowerShell blocks activation, use Command Prompt:

```cmd
cd /d "YOUR\SmartCart_Web"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## What the site includes

- K-Means / Agglomerative clustering switch
- Cluster-count slider from 2–10
- Interactive 3D PCA scatter plot
- Cluster population chart
- WCSS / elbow curve
- Silhouette-score curve
- Cluster summary table
- Customer ID explorer
- Customer-level cluster context
- Responsive dark glass/neon UI

## Important methodology note

The website follows the notebook's main workflow:

1. Fill missing Income values with the median.
2. Engineer Age, Customer_Tenure_Days, Total_Spending, Total_Children.
3. Group Education and Marital_Status into simplified categories.
4. Drop the same raw columns and individual spending columns.
5. Remove Age >= 90 and Income >= 600000.
6. One-hot encode Education and Living_With.
7. Standardize features.
8. Apply 3-component PCA.
9. Run K-Means or Agglomerative clustering on the PCA representation.
10. Visualize the resulting clusters.

The original notebook hard-codes `n_clusters=4` for its final K-Means and Agglomerative models. The website makes this interactive so you can experiment with K values.
