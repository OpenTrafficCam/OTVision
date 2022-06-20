import pandas as pd


def ottrk_dict_to_df(nested_dict):
    return pd.DataFrame.from_dict(
        {
            (i, j): nested_dict[i][j]
            for i in nested_dict.keys()
            for j in nested_dict[i].keys()
        },
        orient="index",
    )


def get_epsg_from_utm_zone(utm_zone, hemisphere):
    identifier_digits = 32000
    if hemisphere == "N":
        hemisphere_digit = 600
    elif hemisphere == "S":
        hemisphere_digit = 700
    return identifier_digits + hemisphere_digit + int(utm_zone)
