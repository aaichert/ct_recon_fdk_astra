from pathlib import Path as _Path


def get_data_path(*relative_parts: str) -> _Path:
    """Return the absolute path to a file inside the installed package data.

    If the requested file does not exist but a ``.zip`` archive with the same
    stem is present in the same directory, the archive is extracted there
    automatically (lazy decompression on first access).  This is the mechanism
    used to unpack the bundled ``fullscan_180views_600x400.nrrd`` on first use.

    Example
    -------
    >>> import ct_recon_fdk_astra as recon
    >>> json_90  = recon.get_data_path("example_data", "fullscan_90views_600x400.json")
    >>> nrrd_90  = recon.get_data_path("example_data", "fullscan_90views_600x400.nrrd")  # auto-extracted
    >>> json_180 = recon.get_data_path("example_data", "fullscan_180views_600x400.json")
    >>> nrrd_180 = recon.get_data_path("example_data", "fullscan_180views_600x400.nrrd")  # auto-extracted

    Parameters
    ----------
    *relative_parts : str
        Path components relative to ``ct_recon_fdk_astra/data/``.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found even after attempting extraction.
    """
    base = _Path(__file__).parent.resolve() / "data"
    result = base.joinpath(*relative_parts)

    if not result.exists():
        # Lazy decompression: look for a .zip archive with the same stem
        # and extract it in-place.
        parent = result.parent
        stem = result.stem
        zip_path = parent / f"{stem}.zip"

        if zip_path.exists():
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(path=parent)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to extract bundled archive {zip_path.name}: {exc}"
                ) from exc

        if not result.exists():
            if zip_path.exists():
                hint = (
                    f"The bundled .zip archive '{zip_path.name}' was extracted "
                    "but the expected file is still missing — "
                    "the archive may contain a different filename."
                )
            else:
                hint = (
                    "No matching .zip archive was found in the same directory. "
                    "The package data may be incomplete."
                )
            raise FileNotFoundError(
                f"Package data not found: {'/'.join(relative_parts)}\n{hint}"
            )

    return result


# ---------------------------------------------------------------------------
# Convenience constants — None when the package data is unavailable.
# ---------------------------------------------------------------------------

try:
    EXAMPLE_DATA_PATH: _Path | None = get_data_path("example_data")
    if EXAMPLE_DATA_PATH is not None:
        get_data_path("example_data", "fullscan_90views_600x400.nrrd")
        get_data_path("example_data", "fullscan_180views_600x400.nrrd")
except FileNotFoundError:
    EXAMPLE_DATA_PATH = None


__all__ = [
    "get_data_path",
    "EXAMPLE_DATA_PATH",
]
