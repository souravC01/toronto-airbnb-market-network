# Data provenance

`toronto_listings_clean.csv` is derived from the **November 2025 Toronto listings snapshot** published by [Inside Airbnb](https://insideairbnb.com/get-the-data/).

## Included snapshot

- 15,809 listings
- 10,326 unique hosts
- 140 official Toronto neighbourhoods
- 24 retained or derived columns

The repository includes the cleaned snapshot so the published network results can be reproduced without relying on a mutable external download.

## Cleaning and derived fields

`src/clean_dataset.py` performs the following transformations:

- Retains the listing, host, location, neighbourhood, room, property, availability, review, and booking fields used by the study
- Removes rows with missing IDs, coordinates, or prices
- Removes duplicate listing IDs and non-positive prices
- Median-imputes selected numerical fields
- Replaces missing categorical values with `Unknown`
- Winsorizes price at the 1st and 99th percentiles
- Creates `price_w`, `log_price`, and `log_price_w`

## Recreate the cleaned file

Download the corresponding raw Toronto listings file from Inside Airbnb, place it in this directory, and run:

```bash
python src/clean_dataset.py \
  --raw data/listings.csv.gz \
  --out data/toronto_listings_clean.csv
```

Raw and compressed downloads are intentionally ignored by Git.

## Use and redistribution

Inside Airbnb publishes data for research and advocacy. Users of this repository should review the current terms and data policies on the source website before redistributing or using the data for another purpose. The repository's future software license will not override any terms that apply to the source data.
