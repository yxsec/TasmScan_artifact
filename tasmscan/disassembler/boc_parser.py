"""BOC (Bag of Cells) format parser"""

from pytoniq_core import Cell


def parse_boc(data: bytes) -> Cell:
    """
    Parse BOC data and return the root cell.

    Args:
        data: BOC data as bytes

    Returns:
        Root Cell object

    Raises:
        ValueError: If BOC data is invalid
    """
    try:
        cells = Cell.one_from_boc(data)
        if cells is None:
            raise ValueError("No cells found in BOC")
        return cells
    except Exception as e:
        raise ValueError(f"Failed to parse BOC: {str(e)}")

