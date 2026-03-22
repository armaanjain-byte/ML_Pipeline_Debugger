from app.pipeline.data_loader import DataLoader
from app.pipeline.preprocessing import Preprocessor
from app.pipeline.model import Model

# Load data
loader = DataLoader("data/bhubaneswar_rent_clean_audited.csv")
df = loader.load_data()

# Preprocess
preprocessor = Preprocessor(target_column="rent")
df_clean = preprocessor.handle_missing_values(df)
df_encoded = preprocessor.encode_categorical(df_clean)

X_train, X_test, y_train, y_test = preprocessor.split_data(df_encoded)

# Model
model = Model(task_type="classification")  # change if needed
model.train(X_train, y_train)

y_pred = model.predict(X_test)

metrics = model.evaluate(y_test, y_pred)

print("Metrics:", metrics)