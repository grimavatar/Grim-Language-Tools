import shutil
from pathlib import Path


def suppress_logging():
    # Suppress standard Python logging warnings
    import logging
    # logging.getLogger().setLevel(logging.ERROR)
    logging.disable(logging.WARNING)

    # # Suppress NeMo's verbose Info (I) and Warning (W) logs
    # import nemo.utils as nemo_utils
    # nemo_utils.logging.setLevel(nemo_utils.logging.ERROR)


def compress(output_path: str | Path) -> None:
    if isinstance(output_path, str):
        output_path = Path(output_path)

    is_empty = not any(output_path.iterdir())
    if is_empty:
        return

    shutil.make_archive(
        base_name = output_path,  # .zip is added automatically
        format = "zip", 
        root_dir = output_path
    )
