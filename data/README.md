# Data Folder

Expected raw dataset path:

- `data/raw/creditcard.csv`

Kaggle source used in this project:

- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download (after `kaggle` CLI auth):

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

This project expects:

- target column: `Class` (1 = fraud, 0 = non-fraud)
- transaction amount column: `Amount`
- relative time column: `Time`
