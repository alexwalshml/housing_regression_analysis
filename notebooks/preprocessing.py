def numeric_object_split(df):
    # split dataset into numerical and object dataframes
    numerics = df.select_dtypes(exclude=[object])
    objects = df.select_dtypes(include=[object])

    return numerics, objects


def z_filter(df, high=3.0, low=-3.0, extras=None):
    # filters data based on any outliers within columns
    z_table = ((df - df.mean(numeric_only=True)) / df.std(numeric_only=True)).fillna(0)
    to_keep = ((z_table < high) & (z_table > low)).all(axis=1)

    if extras:
        extras = [extra[to_keep] for extra in extras]

        return df[to_keep], extras

    return df[to_keep]