from app.pipeline.data_loader import DataLoader
from app.debugger.data_checks import DataChecks

loader = DataLoader("data/bhubaneswar_rent_clean_audited.csv")
df = loader.load_data()

checker = DataChecks(df, target_column="area")

print("Missing:", checker.check_missing_values())
print("Imbalance:", checker.check_class_imbalance())
print("Constant:", checker.check_constant_features())
print("Correlation:", checker.check_high_correlation())