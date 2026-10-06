import pandas as pd
from pathlib import Path


root = Path('data/processed')

print({name: pd.read_csv(root / (name + '.csv'))['label'].value_counts().to_dict() for name in ['train', 'test']})