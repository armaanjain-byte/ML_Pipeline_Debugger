from app.pipeline.data_loader import DataLoader
from app.pipeline.preprocessing import Preprocessor

# Step 1: Load data
loader = DataLoader("data/bhubaneswar_rent_clean_audited.csv")   # change if needed
df = loader.load_data()

# Step 2: Preprocess
preprocessor = Preprocessor(target_column="rent")  # change this

df_clean = preprocessor.handle_missing_values(df)
df_encoded = preprocessor.encode_categorical(df_clean)

X_train, X_test, y_train, y_test = preprocessor.split_data(df_encoded)

print("Shapes:")
print(X_train.shape, X_test.shape)