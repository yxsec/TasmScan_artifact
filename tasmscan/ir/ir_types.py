"""
IR Type Definitions for TVM bytecode analysis.
"""
from enum import Enum


class IRType(Enum):
    """High-level IR instruction types

    Comprehensive coverage of TVM opcodes organized by functionality.
    Based on TVM cp0 specification.
    """

    # ========== Cell Parse Operations (category: cell_parse, 89 instructions) ==========
    # Slice reading - load data from cells
    LOAD_INT = "load_int"                      # LDI, LDIX, LDIQ - Load signed integer
    LOAD_UINT = "load_uint"                    # LDU, LDUX, LDUQ - Load unsigned integer
    LOAD_BITS = "load_bits"                    # LDSLICE, LDSLICEX - Load bit slice
    LOAD_REF = "load_ref"                      # LDREF, LDREFRTOS - Load cell reference
    PRELOAD_INT = "preload_int"                # PLDI, PLDIX - Preload signed int (no advance)
    PRELOAD_UINT = "preload_uint"              # PLDU, PLDUX - Preload unsigned int
    PRELOAD_BITS = "preload_bits"              # PLDSLICE, PLDSLICEX - Preload bit slice
    PRELOAD_REF = "preload_ref"                # PLDREFVAR, PLDREFIDX - Preload cell reference

    # Cell/Slice conversions
    CELL_TO_SLICE = "cell_to_slice"            # CTOS - Convert cell to slice
    END_SLICE = "end_slice"                    # ENDS - Check slice is empty

    # ========== Cell Build Operations (category: cell_build, 65 instructions) ==========
    # Builder writing - construct new cells
    CREATE_CELL_BUILDER = "create_cell_builder"  # NEWC - Create new cell builder
    BUILD_CELL = "build_cell"                    # ENDC - Finalize builder to cell
    STORE_INT = "store_int"                      # STI, STIX, STIQ - Store signed integer
    STORE_UINT = "store_uint"                    # STU, STUX, STUQ - Store unsigned integer
    STORE_BITS = "store_bits"                    # STSLICE, STSLICEX - Store bit slice
    STORE_REF = "store_ref"                      # STREF, STBREFR - Store cell reference

    # ========== Message Operations (High-level abstractions) ==========
    # Message field parsing (context-aware mapping of LOAD_UINT/LOAD_INT)
    LOAD_MESSAGE_FLAGS = "load_message_flags"    # LDU 4 at message start
    LOAD_MESSAGE_SENDER = "load_message_sender"  # LDMSGADDR or LDU 256 after flags
    LOAD_MESSAGE_VALUE = "load_message_value"    # LDU 120 for value (grams)
    LOAD_MESSAGE_BODY = "load_message_body"      # LDREF or LDSLICE for body
    LOAD_MESSAGE_OPCODE = "load_message_opcode"  # LDU 32 for operation code
    CHECK_BOUNCED = "check_bounced"              # AND 1 after flag load

    # ========== Cryptographic Operations (category: app_crypto, 64 instructions) ==========
    HASH_CELL = "hash_cell"                    # HASHCU - Hash cell
    HASH_SLICE = "hash_slice"                  # HASHSU - Hash slice
    SHA256 = "sha256"                          # SHA256U - SHA-256 hash
    CHECK_SIGNATURE = "check_signature"        # CHKSIGNU - Verify Ed25519 signature
    HASH_EXT_SHA256 = "hash_ext_sha256"        # HASHEXT_SHA256
    HASH_EXT_SHA512 = "hash_ext_sha512"        # HASHEXT_SHA512
    HASH_EXT_BLAKE2B = "hash_ext_blake2b"      # HASHEXT_BLAKE2B
    HASH_EXT_KECCAK256 = "hash_ext_keccak256"  # HASHEXT_KECCAK256
    HASH_EXT_KECCAK512 = "hash_ext_keccak512"  # HASHEXT_KECCAK512

    # ========== Stack Operations (categories: stack_basic, stack_complex) ==========
    # Basic stack
    PUSH_VALUE = "push_value"                  # PUSH, PUSH2, PUSH3, PUSH_LONG
    POP_VALUE = "pop_value"                    # POP, POP_LONG, DROP, DROP2
    DUP_VALUE = "dup_value"                    # DUP, DUP2, OVER

    # Complex stack manipulation
    STACK_EXCHANGE = "stack_exchange"          # XCHG, XCHG_0I, XCHG_IJ, XCHG2, XCHG3
    STACK_COMPLEX = "stack_complex"            # XCPU, XC2PU, PUXC, XCPUXC, etc.
    STACK_ROTATE = "stack_rotate"              # ROT, ROTREV
    STACK_SWAP = "stack_swap"                  # SWAP, SWAP2
    STACK_BLOCK_SWAP = "stack_block_swap"      # BLKSWAP - Swap stack blocks
    STACK_REVERSE = "stack_reverse"            # REVERSE - Reverse stack segment
    STACK_PICK = "stack_pick"                  # PICK - Copy si to top
    STACK_ROLL = "stack_roll"                  # ROLL - Move si to top

    # ========== Control Flow - Basic (category: cont_basic, 22 instructions) ==========
    CALL = "call"                              # CALLREF, CALLXARGS
    CALL_INDIRECT = "call_indirect"            # EXECUTE, JMPX
    RETURN = "return"                          # RET, RETALT, RETARGS
    UNCONDITIONAL_JUMP = "unconditional_jump"  # JMPREF, JMPX, JMPXDATA

    # ========== Control Flow - Conditional (category: cont_conditional, 22 instructions) ==========
    CONDITIONAL_BRANCH = "conditional_branch"  # IF, IFNOT, IFELSE
    CONDITIONAL_JUMP = "conditional_jump"      # IFJMP, IFNOTJMP, IFJMPREF
    CONDITIONAL_RETURN = "conditional_return"  # IFRET, IFNOTRET, IFRETALT
    SELECT_CONDITIONAL = "select_conditional"  # CONDSEL, CONDSELCHK
    BIT_CONDITIONAL_JUMP = "bit_conditional_jump"  # IFBITJMP, IFNBITJMP

    # ========== Control Flow - Loops (category: cont_loops, 16 instructions) ==========
    LOOP_REPEAT = "loop_repeat"                # REPEAT, REPEATEND, REPEATBRK
    LOOP_UNTIL = "loop_until"                  # UNTIL, UNTILEND, UNTILBRK
    LOOP_WHILE = "loop_while"                  # WHILE, WHILEEND, WHILEBRK
    LOOP_AGAIN = "loop_again"                  # AGAIN, AGAINEND, AGAINBRK (infinite)

    # ========== Control Flow - Registers (category: cont_registers, 24 instructions) ==========
    PUSH_CONTINUATION = "push_continuation"    # PUSHCTR, PUSHCONT - Push continuation
    POP_CONTINUATION = "pop_continuation"      # POPCTR - Pop continuation
    SET_CONTINUATION = "set_continuation"      # SETCONTCTR, SETRETCTR, SETALTCTR
    SAVE_CONTINUATION = "save_continuation"    # SAVE, SAVEALT, SAVEBOTH, POPSAVE

    # ========== Storage Operations ==========
    LOAD_STORAGE = "load_storage"              # PUSHCTR c4 - Load persistent storage
    STORE_STORAGE = "store_storage"            # POPCTR c4 - Save persistent storage

    # ========== Exception Handling (category: exceptions, 17 instructions) ==========
    THROW = "throw"                            # THROW, THROW_SHORT, THROWARG
    THROW_IF = "throw_if"                      # THROWIF, THROWIF_SHORT, THROWARGIF
    THROW_IF_NOT = "throw_if_not"              # THROWIFNOT, THROWIFNOT_SHORT
    THROW_ANY = "throw_any"                    # THROWANY, THROWANYIF, THROWANYIFNOT
    TRY_CATCH = "try_catch"                    # TRY, TRYARGS

    # ========== Arithmetic Operations (categories: arithm_*) ==========
    # Basic arithmetic
    ARITHMETIC = "arithmetic"                  # ADD, SUB, MUL, INC, DEC
    ARITHMETIC_DIV = "arithmetic_div"          # DIV, MOD, DIVMOD (94 variants)
    ARITHMETIC_SHIFT = "arithmetic_shift"      # LSHIFT, RSHIFT, POW2
    ARITHMETIC_NEGATE = "arithmetic_negate"    # NEG, ABS

    # Quiet arithmetic (no overflow exceptions)
    ARITHMETIC_QUIET = "arithmetic_quiet"      # QADD, QSUB, QMUL, QDIV (114 variants)

    # ========== Comparison Operations ==========
    COMPARISON = "comparison"                  # EQUAL, LESS, GREATER, LEQ, GEQ, NEQ
    COMPARISON_SGN = "comparison_sgn"          # SGN, ISPOS, ISNEG, ISZERO, ISNAN

    # ========== Logical Operations ==========
    BITWISE = "bitwise"                        # AND, OR, XOR, NOT
    BITWISE_SHIFT = "bitwise_shift"            # LSHIFT, RSHIFT

    # ========== Slice Manipulation Operations ==========
    SLICE_CUT_FIRST = "slice_cut_first"        # SDCUTFIRST, SCUTFIRST - Cut first bits
    SLICE_SKIP_FIRST = "slice_skip_first"      # SDSKIPFIRST, SSKIPFIRST - Skip first bits
    SLICE_CUT_LAST = "slice_cut_last"          # SDCUTLAST, SCUTLAST - Cut last bits
    SLICE_SKIP_LAST = "slice_skip_last"        # SDSKIPLAST, SSKIPLAST - Skip last bits
    SLICE_SPLIT = "slice_split"                # SPLIT, SPLITQ - Split slice
    SLICE_INFO = "slice_info"                  # SBITS, SREFS, SBITREFS - Query slice info
    CELL_INFO = "cell_info"                    # CDEPTH, CLEVEL, CHASHI, CDEPTHI - Query cell info

    # ========== Builder Info Operations ==========
    BUILDER_INFO = "builder_info"              # BBITS, BREFS, BBITREFS - Query builder info
    BUILDER_REMAINING = "builder_remaining"    # BREMBITS, BREMREFS, BREMBITREFS - Remaining space

    # ========== Little-Endian Integer Operations ==========
    LOAD_INT_LE = "load_int_le"                # LDILE4, LDILE8 - Load little-endian signed
    LOAD_UINT_LE = "load_uint_le"              # LDULE4, LDULE8 - Load little-endian unsigned
    PRELOAD_INT_LE = "preload_int_le"          # PLDILE4, PLDILE8 - Preload little-endian signed
    PRELOAD_UINT_LE = "preload_uint_le"        # PLDULE4, PLDULE8 - Preload little-endian unsigned
    STORE_INT_LE = "store_int_le"              # STILE4, STILE8 - Store little-endian signed
    STORE_UINT_LE = "store_uint_le"            # STULE4, STULE8 - Store little-endian unsigned

    # ========== Optional Reference Operations ==========
    LOAD_OPT_REF = "load_opt_ref"              # LDOPTREF - Load optional reference
    STORE_OPT_REF = "store_opt_ref"            # STOPTREF - Store optional reference

    # ========== Tuple Operations (category: tuple, 34 instructions) ==========
    TUPLE_CREATE = "tuple_create"              # TUPLE, TUPLEVAR
    TUPLE_INDEX = "tuple_index"                # INDEX, INDEXVAR
    TUPLE_SET = "tuple_set"                    # SETINDEX, SETINDEXVAR
    TUPLE_LENGTH = "tuple_length"              # TLEN
    TUPLE_OPERATIONS = "tuple_operations"      # TPUSH, TPOP, etc.

    # ========== Dictionary Operations (categories: dict_*) ==========
    DICT_GET = "dict_get"                      # DICTGET, DICTIGET, DICTUGET
    DICT_SET = "dict_set"                      # DICTSET, DICTISET, DICTUSET
    DICT_DELETE = "dict_delete"                # DICTDEL, DICTIDEL, DICTUDEL
    DICT_ITERATE = "dict_iterate"              # DICTNEXT, DICTPREV, DICTMIN, DICTMAX
    DICT_OPERATIONS = "dict_operations"        # Generic dict ops
    DICT_LOAD = "dict_load"                    # LDDICT - Load dictionary
    DICT_STORE = "dict_store"                  # STDICT - Store dictionary
    DICT_EMPTY = "dict_empty"                  # DICTEMPTY - Check if dict is empty
    DICT_GET_OPT = "dict_get_opt"              # DICTGETOPTREF - Get optional from dict

    # ========== Application-Level: Gas Operations (category: app_gas, 4 instructions) ==========
    ACCEPT_GAS = "accept_gas"                  # ACCEPT - Accept gas payment
    SET_GAS_LIMIT = "set_gas_limit"            # SETGASLIMIT - Set gas limit
    GET_GAS_CONSUMED = "get_gas_consumed"      # GASCONSUMED - Get gas used
    COMMIT_STATE = "commit_state"              # COMMIT - Commit state changes

    # ========== Application-Level: Actions (category: app_actions, 7 instructions) ==========
    SEND_MESSAGE = "send_message"              # SENDRAWMSG, SENDMSG - Send message
    RESERVE_CURRENCY = "reserve_currency"      # RAWRESERVE, RAWRESERVEX - Reserve balance
    SET_CODE = "set_code"                      # SETCODE - Update contract code
    SET_LIB_CODE = "set_lib_code"              # SETLIBCODE, CHANGELIB - Manage libraries

    # ========== Application-Level: Address Operations (category: app_addr, 8 instructions) ==========
    LOAD_MSG_ADDRESS = "load_msg_address"      # LDMSGADDR, LDMSGADDRQ - Load address
    PARSE_MSG_ADDRESS = "parse_msg_address"    # PARSEMSGADDR - Parse address
    REWRITE_ADDRESS = "rewrite_address"        # REWRITESTDADDR, REWRITEVARADDR

    # ========== Application-Level: Config/Blockchain (category: app_config, 14 instructions) ==========
    GET_BLOCKCHAIN_PARAM = "get_blockchain_param"  # GETPARAM - Get blockchain parameter
    GET_CONFIG_DICT = "get_config_dict"            # CONFIGDICT - Get config dictionary
    GET_CONFIG_PARAM = "get_config_param"          # CONFIGPARAM, CONFIGOPTPARAM
    GET_GLOBAL_ID = "get_global_id"                # GLOBALID - Get global chain ID
    GET_BLOCK_INFO = "get_block_info"              # PREVMCBLOCKS, PREVKEYBLOCK
    GET_GAS_FEE = "get_gas_fee"                    # GETGASFEE, GETGASFEESIMPLE
    GET_STORAGE_FEE = "get_storage_fee"            # GETSTORAGEFEE
    GET_FORWARD_FEE = "get_forward_fee"            # GETFORWARDFEE, GETFORWARDFEESIMPLE

    # ========== Application-Level: Currency (category: app_currency, 8 instructions) ==========
    LOAD_GRAMS = "load_grams"                  # LDGRAMS, LDVARINT16 - Load currency
    STORE_GRAMS = "store_grams"                # STGRAMS, STVARINT16 - Store currency
    LOAD_VARINT = "load_varint"                # LDVARINT32, LDVARUINT32
    STORE_VARINT = "store_varint"              # STVARINT32, STVARUINT32

    # ========== Application-Level: Global Variables (category: app_global, 4 instructions) ==========
    GET_GLOBAL_VAR = "get_global_var"          # GETGLOBVAR, GETGLOB
    SET_GLOBAL_VAR = "set_global_var"          # SETGLOBVAR, SETGLOB

    # ========== Application-Level: Misc (category: app_misc, 4 instructions) ==========
    GET_CELL_DATA_SIZE = "get_cell_data_size"  # CDATASIZE, CDATASIZEQ
    GET_SLICE_DATA_SIZE = "get_slice_data_size"  # SDATASIZE, SDATASIZEQ

    # ========== Extended Tuple Operations (additions to existing tuple ops) ==========
    TUPLE_NULL = "tuple_null"                  # NULL - Push null tuple
    TUPLE_IS_NULL = "tuple_is_null"            # ISNULL - Check if null
    TUPLE_IS_TUPLE = "tuple_is_tuple"          # ISTUPLE - Check if tuple
    TUPLE_UNPACK = "tuple_unpack"              # UNTUPLE, UNTUPLEVAR - Unpack to elements
    TUPLE_UNPACK_FIRST = "tuple_unpack_first"  # UNPACKFIRST, UNPACKFIRSTVAR
    TUPLE_EXPLODE = "tuple_explode"            # EXPLODE, EXPLODEVAR - Unpack with length
    TUPLE_LAST = "tuple_last"                  # LAST - Get last element

    # ========== Constant Operations (category: const_int, const_data) ==========
    PUSH_INT = "push_int"                      # PUSHINT, PUSHINT_4, PUSHINT_8, PUSHINT_16, PUSHINT_LONG
    PUSH_NAN = "push_nan"                      # PUSHNAN - Push Not-a-Number
    PUSH_REF = "push_ref"                      # PUSHREF - Push reference
    PUSH_SLICE = "push_slice"                  # PUSHSLICE, PUSHSLICE_REFS - Push slice constant
    PUSH_CONT = "push_cont"                    # PUSHCONT, PUSHREFCONT - Push continuation

    # ========== Stack Complex Operations (category: stack_complex) ==========
    STACK_PUSH2 = "stack_push2"                # PUSH2, OVER2 - Push 2 elements
    STACK_XCHG = "stack_xchg"                  # XCHG, XCHGX - Exchange stack elements
    STACK_DEPTH = "stack_depth"                # DEPTH - Get stack depth
    STACK_CHECK_DEPTH = "stack_check_depth"    # CHKDEPTH - Check stack depth
    STACK_ONLY_TOP = "stack_only_top"          # ONLYTOPX, ONLYX - Keep only top elements
    STACK_BLK_PUSH = "stack_blk_push"          # BLKPUSH - Block push
    STACK_BLK_DROP = "stack_blk_drop"          # BLKDROP - Block drop
    STACK_BLK_SWAP = "stack_blk_swap"          # BLKSWAP - Block swap
    STACK_TUCK = "stack_tuck"                  # TUCK - Tuck element

    # ========== Arithmetic Division Operations (category: arithm_div) ==========
    DIV_MOD = "div_mod"                        # DIVMOD, DIVMODR, DIVMODC
    MUL_DIV = "mul_div"                        # MULDIV, MULDIVR, MULDIVC
    MUL_MOD = "mul_mod"                        # MULMOD, MULMODR, MULMODC
    ADD_DIV_MOD = "add_div_mod"                # ADDDIVMOD, ADDDIVMODR, ADDDIVMODC
    MOD_ROUND = "mod_round"                    # MODR, RSHIFTR - Modulo with rounding

    # ========== Arithmetic Quiet Operations (category: arithm_quiet) ==========
    # Quiet operations return NaN on errors instead of throwing
    QUIET_FIT = "quiet_fit"                    # QFITS, QFITSX, QUFITS, QUFITSX
    QUIET_ADD = "quiet_add"                    # QADD, QSUB, etc.
    QUIET_MUL = "quiet_mul"                    # QMUL, QDIV, etc.

    # ========== Comparison Operations (category: compare_other) ==========
    COMPARE_SLICE_EQ = "compare_slice_eq"      # SDEQ - Slice data equality
    COMPARE_SLICE_PREFIX = "compare_slice_prefix"  # SDPFX, SDPPFX - Slice prefix check

    # ========== Advanced Crypto (category: app_crypto) ==========
    # BLS12-381 Curve Operations
    CRYPTO_BLS_G1_ADD = "crypto_bls_g1_add"    # BLS_G1_ADD, etc.
    CRYPTO_BLS_G1_MUL = "crypto_bls_g1_mul"
    CRYPTO_BLS_PAIRING = "crypto_bls_pairing"  # BLS_PAIRING
    # secp256r1 (P-256) Operations
    CRYPTO_P256_CHKSIG = "crypto_p256_chksig"  # P256_CHKSIGNU, P256_CHKSIGNS
    # Rist255 Curve Operations
    CRYPTO_RIST255_OP = "crypto_rist255_op"    # RIST255_ADD, SUB, MUL, etc.
    # Other crypto
    CRYPTO_ECRECOVER = "crypto_ecrecover"      # ECRECOVER - Recover public key

    # ========== Debug/Other ==========
    DEBUG_PRINT = "debug_print"                # DEBUGSTR, DUMP, etc.
    NOP = "nop"                                # NOP - No operation

    # Fallback for unmapped opcodes
    UNKNOWN = "unknown"
    RAW_OPCODE = "raw_opcode"

