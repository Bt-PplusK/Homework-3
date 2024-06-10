import requests
from io import BytesIO
from typing import List
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:

    year = 2023
    month = 3
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for {year}-{month:02d}: {response.text}")

    df = pd.read_parquet(BytesIO(response.content))
    return df



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'