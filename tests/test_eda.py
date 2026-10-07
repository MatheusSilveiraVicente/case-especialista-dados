from src.eda import titulo_preco_desconto


def test_titulo_preco_desconto_escapa_cifrao() -> None:
    titulo = titulo_preco_desconto()
    assert r"R\$ 59" in titulo
    assert r"R\$ 93" in titulo
