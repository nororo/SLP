# Code For SLP Shopping Matching and SLP Second-Order Proximity


### 1. Download dunnhumby dataset

Download the dataset from the Dunnhumby website:
https://www.dunnhumby.com/source-files/

And save it to the `data` directory.

Dataset structure:
```
data/
├── transaction_data.csv
├── hh_demographic.csv
├── campaign_table.csv
├── campaign_desc.csv
├── product.csv
├── coupon.csv
├── coupon_redempt.csv
├── causal_data.csv
```


### 2. Preprocess the dataset

Run the `preprocess_dataset.py` script to preprocess the dataset.
```
python preprocess_data.py
```

This will create the following files in the data directory:
```
data/
├── proc_data_transaction.parquet
```

### 3. Train shopping embedding model

Run the `slp/shopping_embedding.py` script to train the shopping embedding model.
```
python slp/shopping_embedding.py
```


### 4. Inference shopping embedding features

Run the `slp/inference_shopping_embedding.py` script to inference the shopping embedding features.
```
python slp/inference_shopping_embedding.py
```

This will create the following files in the data directory:
```
data/
├── features.parquet
```

### 5. Generate substitute pairs

Run the `process_data.py` script to generate the substitute pairs.
```
python process_data.py
```

This will create substitute_pairs files in the data/method_name directory:
```
data/
├── {method_name}
│   ├── substitute_pairs.parquet
```

