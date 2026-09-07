"""
Cell Operations - Parse and Build Operations

This module contains mappings for:
- Cell parse operations (LDI, LDU, LDREF, etc.)
- Cell build operations (NEWC, STI, STU, STREF, etc.)
- Slice/Builder manipulation and info
"""

from ..ir_types import IRType

# Cell Parse Operations
CELL_PARSE_MAPPINGS = {
    # ========== Cell Parse Operations ==========
    # Slice reading - load data from cells
    "LDI": IRType.LOAD_INT,
    "LDIX": IRType.LOAD_INT,
    "LDIQ": IRType.LOAD_INT,
    "LDI_ALT": IRType.LOAD_INT,
    "LDIXQ": IRType.LOAD_INT,

    "LDU": IRType.LOAD_UINT,
    "LDUX": IRType.LOAD_UINT,
    "LDUQ": IRType.LOAD_UINT,
    "LDU_ALT": IRType.LOAD_UINT,
    "LDUXQ": IRType.LOAD_UINT,
    "PLDUZ": IRType.LOAD_UINT,

    "LDSLICE": IRType.LOAD_BITS,
    "LDSLICEX": IRType.LOAD_BITS,
    "LDSLICEXQ": IRType.LOAD_BITS,
    "LDSLICE_ALT": IRType.LOAD_BITS,

    "LDREF": IRType.LOAD_REF,
    "LDREFRTOS": IRType.LOAD_REF,
    "LDREFIDX": IRType.LOAD_REF,

    # Preload (don't advance slice position)
    "PLDI": IRType.PRELOAD_INT,
    "PLDIX": IRType.PRELOAD_INT,
    "PLDIQ": IRType.PRELOAD_INT,
    "PLDIXQ": IRType.PRELOAD_INT,

    "PLDU": IRType.PRELOAD_UINT,
    "PLDUX": IRType.PRELOAD_UINT,
    "PLDUQ": IRType.PRELOAD_UINT,
    "PLDUXQ": IRType.PRELOAD_UINT,

    "PLDSLICE": IRType.PRELOAD_BITS,
    "PLDSLICEX": IRType.PRELOAD_BITS,
    "PLDSLICEXQ": IRType.PRELOAD_BITS,

    # Cell/Slice conversions
    "CTOS": IRType.CELL_TO_SLICE,
    "ENDS": IRType.END_SLICE,

    # Slice manipulation
    "SDCUTFIRST": IRType.SLICE_CUT_FIRST,
    "SCUTFIRST": IRType.SLICE_CUT_FIRST,
    "SDSKIPFIRST": IRType.SLICE_SKIP_FIRST,
    "SSKIPFIRST": IRType.SLICE_SKIP_FIRST,
    "SDCUTLAST": IRType.SLICE_CUT_LAST,
    "SCUTLAST": IRType.SLICE_CUT_LAST,
    "SDSKIPLAST": IRType.SLICE_SKIP_LAST,
    "SSKIPLAST": IRType.SLICE_SKIP_LAST,
    "SPLIT": IRType.SLICE_SPLIT,
    "SPLITQ": IRType.SLICE_SPLIT,
    "SDSUBSTR": IRType.SLICE_SPLIT,
    "SUBSLICE": IRType.SLICE_SPLIT,

    # Slice info
    "SBITS": IRType.SLICE_INFO,
    "SREFS": IRType.SLICE_INFO,
    "SBITREFS": IRType.SLICE_INFO,
    "SREMBITREFS": IRType.SLICE_INFO,
    "SCHKBITS": IRType.SLICE_INFO,
    "SCHKREFS": IRType.SLICE_INFO,
    "SCHKBITREFS": IRType.SLICE_INFO,

    # Little-endian integer loading
    "LDILE4": IRType.LOAD_INT_LE,
    "LDILE8": IRType.LOAD_INT_LE,
    "LDULE4": IRType.LOAD_UINT_LE,
    "LDULE8": IRType.LOAD_UINT_LE,
    "PLDILE4": IRType.PRELOAD_INT_LE,
    "PLDILE8": IRType.PRELOAD_INT_LE,
    "PLDULE4": IRType.PRELOAD_UINT_LE,
    "PLDULE8": IRType.PRELOAD_UINT_LE,

    # Optional reference
    "LDOPTREF": IRType.LOAD_OPT_REF,

    # Cell Parse Quiet Operations
    "LDSLICEQ": IRType.LOAD_BITS,
    "PLDSLICEQ": IRType.PRELOAD_BITS,
    "SCHKBITSQ": IRType.SLICE_INFO,
    "SCHKREFSQ": IRType.SLICE_INFO,
    "SCHKBITREFSQ": IRType.SLICE_INFO,
    "SDBEGINSX": IRType.SLICE_INFO,
    "SDBEGINSXQ": IRType.SLICE_INFO,
    "SDBEGINS": IRType.SLICE_INFO,
    "SDBEGINSQ": IRType.SLICE_INFO,
    "XCTOS": IRType.LOAD_BITS,
    "XLOAD": IRType.LOAD_REF,
    "XLOADQ": IRType.LOAD_REF,
    "ENDXC": IRType.BUILD_CELL,

    # Additional cell_parse operations
    "PLDREFVAR": IRType.PRELOAD_REF,
    "PLDREFIDX": IRType.PRELOAD_REF,
    "LDILE4Q": IRType.LOAD_INT,
    "LDULE4Q": IRType.LOAD_UINT,
    "LDILE8Q": IRType.LOAD_INT,
    "LDULE8Q": IRType.LOAD_UINT,
    "PLDILE4Q": IRType.PRELOAD_INT,
    "PLDULE4Q": IRType.PRELOAD_UINT,
    "PLDILE8Q": IRType.PRELOAD_INT,
    "PLDULE8Q": IRType.PRELOAD_UINT,
    "LDZEROES": IRType.LOAD_BITS,
    "LDONES": IRType.LOAD_BITS,
    "LDSAME": IRType.LOAD_BITS,
    "SDEPTH": IRType.SLICE_INFO,
    "CDEPTH": IRType.CELL_INFO,
    "CLEVEL": IRType.CELL_INFO,
    "CLEVELMASK": IRType.CELL_INFO,
    "CHASHI": IRType.CELL_INFO,
    "CDEPTHI": IRType.CELL_INFO,
    "CHASHIX": IRType.CELL_INFO,
    "CDEPTHIX": IRType.CELL_INFO,

    # Slice comparisons (compare_other)
    "SDCNTLEAD0": IRType.SLICE_INFO,
    "SDCNTLEAD1": IRType.SLICE_INFO,
    "SDCNTTRAIL0": IRType.SLICE_INFO,
    "SDCNTTRAIL1": IRType.SLICE_INFO,
    "SDEMPTY": IRType.SLICE_INFO,
    "SDFIRST": IRType.SLICE_INFO,
    "SEMPTY": IRType.SLICE_INFO,
    "SREMPTY": IRType.SLICE_INFO,
}

# Cell Build Operations
CELL_BUILD_MAPPINGS = {
    # ========== Cell Build Operations ==========
    "NEWC": IRType.CREATE_CELL_BUILDER,
    "ENDC": IRType.BUILD_CELL,
    "ENDCST": IRType.BUILD_CELL,

    "STI": IRType.STORE_INT,
    "STIX": IRType.STORE_INT,
    "STIQ": IRType.STORE_INT,
    "STI_ALT": IRType.STORE_INT,
    "STIR": IRType.STORE_INT,
    "STIXR": IRType.STORE_INT,
    "STIXQ": IRType.STORE_INT,
    "STIXRQ": IRType.STORE_INT,

    "STU": IRType.STORE_UINT,
    "STUX": IRType.STORE_UINT,
    "STUQ": IRType.STORE_UINT,
    "STU_ALT": IRType.STORE_UINT,
    "STUR": IRType.STORE_UINT,
    "STUXR": IRType.STORE_UINT,
    "STUXQ": IRType.STORE_UINT,
    "STUXRQ": IRType.STORE_UINT,

    "STSLICE": IRType.STORE_BITS,
    "STSLICEX": IRType.STORE_BITS,
    "STSLICEQ": IRType.STORE_BITS,
    "STSLICEXQ": IRType.STORE_BITS,

    "STREF": IRType.STORE_REF,
    "STBREF": IRType.STORE_REF,
    "STBREFR": IRType.STORE_REF,
    "STREFR": IRType.STORE_REF,
    "STREFQ": IRType.STORE_REF,
    "STBREFQ": IRType.STORE_REF,
    "STREFRQ": IRType.STORE_REF,
    "STBREFRQ": IRType.STORE_REF,

    # Little-endian integer storing
    "STILE4": IRType.STORE_INT_LE,
    "STILE8": IRType.STORE_INT_LE,
    "STULE4": IRType.STORE_UINT_LE,
    "STULE8": IRType.STORE_UINT_LE,

    # Optional reference storing
    "STOPTREF": IRType.STORE_OPT_REF,

    # Builder info
    "BBITS": IRType.BUILDER_INFO,
    "BREFS": IRType.BUILDER_INFO,
    "BBITREFS": IRType.BUILDER_INFO,
    "BREMBITS": IRType.BUILDER_REMAINING,
    "BREMREFS": IRType.BUILDER_REMAINING,
    "BREMBITREFS": IRType.BUILDER_REMAINING,

    # Cell Build Quiet Operations
    "STIRQ": IRType.STORE_INT,
    "STURQ": IRType.STORE_UINT,
    "STBQ": IRType.STORE_BITS,
    "STSLICERQ": IRType.STORE_BITS,
    "STB": IRType.STORE_BITS,
    "STREF_ALT": IRType.STORE_REF,
    "STSLICE_ALT": IRType.STORE_BITS,
    "STBREFR_ALT": IRType.STORE_REF,
    "STSLICER": IRType.STORE_BITS,
    "STBR": IRType.STORE_BITS,
    "STBRQ": IRType.STORE_BITS,

    # Additional cell_build operations
    "STREFCONST": IRType.STORE_REF,
    "STREF2CONST": IRType.STORE_REF,
    "STSLICECONST": IRType.STORE_BITS,
    "STZERO": IRType.STORE_BITS,
    "STSAME": IRType.STORE_BITS,
    "STONES": IRType.STORE_BITS,
    "STZEROES": IRType.STORE_BITS,
    "BDEPTH": IRType.BUILDER_INFO,
    "BCHKBITS": IRType.BUILDER_INFO,
    "BCHKBITS_VAR": IRType.BUILDER_INFO,
    "BCHKREFS": IRType.BUILDER_INFO,
    "BCHKBITREFS": IRType.BUILDER_INFO,
    "BCHKBITSQ": IRType.BUILDER_INFO,
    "BCHKBITSQ_VAR": IRType.BUILDER_INFO,
    "BCHKREFSQ": IRType.BUILDER_INFO,
    "BCHKBITREFSQ": IRType.BUILDER_INFO,

    # Builder/Slice conversions
    "BTOS": IRType.BUILD_CELL,
    "STOC": IRType.BUILD_CELL,
}

# Combined cell operations
CELL_OPS_MAPPINGS = {**CELL_PARSE_MAPPINGS, **CELL_BUILD_MAPPINGS}
