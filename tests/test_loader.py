from app.pipeline.data_loader import DataLoader

# give path to any CSV file you have
file_path = "data/bhubaneswar_rent_clean_audited.csv"   # change this

loader = DataLoader(file_path)

# load dataset
df = loader.load_data()

# get basic info
info = loader.basic_info(df)

print("DATA LOADED SUCCESSFULLY\n")
print(info)