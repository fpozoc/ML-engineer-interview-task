import pandas as pd
import pytest

from ..src.utils.utils import remove_outliers_by_quantile

DF_EXPECTED = pd.DataFrame(
    {
        "telekomUploadSpeed": [40.0, 40.0, 2.4, 40.0],
        "totalRent": [693.16, 1545.8, 226.0, 425.0],
        "yearConstructed": [1967.0, 2018.0, 1973.0, 1928000.0],
    }
)

DF_UNEXPECTED_1 = pd.DataFrame(
    {
        "telekomUploadSpeed": [40.0, 40.0, 2.4, 40.0],
        "totalRent": [693.16, 1545.8, 226.0, 425.0],
        "test": [1967.0, 2018.0, 1973.0, 1928.0],
    }
)

DF_UNEXPECTED_2 = pd.DataFrame(
    {
        "telekomUploadSpeed": [40.0, None, 2.4, 40.0],
        "totalRent": [693.16, 1545.8, 226.0, 425.0],
        "yearConstructed": [1967.0, 2018.0, 1973.0, 1928.0],
    }
)


@pytest.mark.parametrize(
    "df, error_expected",
    [
        (DF_EXPECTED, False),
        (DF_UNEXPECTED_1, True),
        (DF_UNEXPECTED_2, True),
    ],
)
def test_remove_outliers_by_quantile(df: pd.DataFrame, error_expected: bool) -> None:
    """Tests whether outlier removal creates unexpected behaviors."""
    try:
        df_outlier_removed = remove_outliers_by_quantile(df, col="yearConstructed")
        assert df.shape[0] > 0  # Rows
        assert df.shape[1] == df_outlier_removed.shape[1]
        assert df.shape[0] > df_outlier_removed.shape[0]
    except KeyError:
        assert error_expected
    except ValueError:
        assert error_expected
