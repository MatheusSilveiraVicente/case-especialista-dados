import pandas as pd
import pytest


@pytest.fixture
def vendas_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dt_hr": pd.to_datetime(
                [
                    "2025-11-01 00:00:00",
                    "2025-11-01 00:00:00",
                    "2025-11-01 01:00:00",
                    "2025-11-01 02:00:00",
                    "2025-11-02 00:00:00",
                    "2025-11-02 01:00:00",
                    "2025-11-03 23:00:00",
                ]
            ),
            "canal": ["Site", "Site", "App", "Site", "App", "Site", "App"],
            "categoria": [
                "0.111",
                "0.222",
                "Perfumaria",
                "Maquiagem",
                "Perfumaria",
                "Maquiagem",
                "Maquiagem",
            ],
            "receita": [10.0, 20.0, -5.0, 50.0, 0.0, 40.0, 70.0],
            "pedidos": [1, 2, 1, 2, 0, 0, 7],
            "itens": [1, 2, 1, 5, 0, 0, 10],
            "desconto": [1.0, 2.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        }
    )
