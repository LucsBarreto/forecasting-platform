"""
utilitários para manipulação de arquivos.

fornece funções auxiliares para localizar arquivos de entrada.
"""

from __future__ import annotations

from pathlib import Path


def find_files(
    input_path: str | Path,
    pattern: str = "*",
) -> list[Path]:
    """
    localiza arquivos que correspondem a um padrão em um diretório.

    parameters
    ----------
    input_path
        diretório que contém os arquivos de entrada.

    pattern
        padrão glob usado para filtrar os arquivos.

    returns
    -------
    list[path]
        lista ordenada dos arquivos encontrados.

    raises
    ------
    filenotfounderror
        se o diretório informado não existir.

    notadirectoryerror
        se input_path não for um diretório.
    """

    directory = Path(input_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"diretório não encontrado: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"o caminho informado não é um diretório: {directory}"
        )

    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file()
    )