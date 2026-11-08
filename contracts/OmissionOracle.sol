// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

/// @title OmissionOracle
/// @notice Consortium-wide omission check for interbank audit evidence. Every member bank publishes, per epoch, a
///         masked linear sketch (an invertible Bloom lookup table) of the interbank transfers in its books: +1 for
///         transfers it paid, -1 for transfers it received. Pairwise masks cancel in the sum, and so does every
///         transfer that both witnesses recorded identically, so the sum of all sketches holds only the
///         discrepancies (omitted, fabricated or altered transfers). Anyone can read the sum and decode it.
///
///         An epoch runs in phases, each closed by a barrier (every member acted, or the phase window expired):
///           1. books:   each member commits to the Merkle root of its epoch book (commitBooks);
///           2. seed:    the seed is keccak(hash of the block after the books barrier, epoch); it keys the sketch
///                       hash functions and the mask graph, so no bank chose its entries knowing them;
///           3. sketch:  each member that committed books commits to keccak(sketch) (commitSketch);
///           4. reveal:  only after the sketch barrier, members reveal sketches that match their commitments,
///                       within the reveal window (reveal or accumulate); late sketches are rejected. For a member
///                       that did not reveal, its mask neighbours post the masks they shared (revealMask).
///         Commit mode stops after phase 3: sketches are exchanged off chain and one member posts the sum.
///
///         Cell layout: two 256-bit words per cell. Word 0 is the XOR of the 256-bit keys. Word 1 packs a signed
///         count in bits 128..191 (added modulo 2^64) and a 128-bit check value in bits 0..127 (XOR).
contract OmissionOracle {
    address public immutable admin;
    uint256 public immutable cells;          // m, cells per sketch
    uint256 public immutable window;         // blocks per phase before a barrier can be forced

    address[] public members;
    mapping(address => uint256) public memberIndex;   // 1-based
    mapping(address => bytes32) public agreementKey;  // X25519 public key used to derive pairwise keys

    struct Epoch {
        uint64 booksOpened;     // block of the first book commitment
        uint64 booksDone;       // block that closed the books phase
        uint64 sketchesDone;    // block that closed the sketch phase
        uint32 bookCount;
        uint32 sketchCount;
        uint32 revealCount;
        bytes32 seed;
    }

    mapping(uint256 => Epoch) public epochs;
    mapping(uint256 => mapping(address => bytes32)) public bookRoot;      // epoch => member => Merkle root
    mapping(uint256 => mapping(address => bytes32)) public sketchHash;    // epoch => member => keccak(sketch)
    mapping(uint256 => mapping(address => bool)) public revealed;
    mapping(uint256 => mapping(uint256 => uint256)) private acc;          // epoch => word => value
    mapping(uint256 => bytes32) public aggregateHash;                     // commit mode: hash of the posted sum

    uint256 private constant LOW64 = 0xFFFFFFFFFFFFFFFF;
    uint256 private constant LOW128 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;

    event MemberAdded(address indexed bank, bytes32 agreementKey);
    event BooksCommitted(uint256 indexed epoch, address indexed bank, bytes32 root);
    event SeedDrawn(uint256 indexed epoch, bytes32 seed);
    event SketchCommitted(uint256 indexed epoch, address indexed bank, bytes32 sketchHash);
    event SketchRevealed(uint256 indexed epoch, address indexed bank, bool accumulated);
    event MaskRevealed(uint256 indexed epoch, address indexed absent, address indexed neighbour);
    event AggregatePosted(uint256 indexed epoch, address indexed poster, bytes32 aggregateHash);

    error NotAdmin();
    error NotMember();
    error Twice();
    error BadLength();
    error WrongPhase();
    error NoCommitment();
    error Mismatch();

    constructor(uint256 m, uint256 phaseWindow) {
        admin = msg.sender;
        cells = m;
        window = phaseWindow;
    }

    function addMembers(address[] calldata banks, bytes32[] calldata keys) external {
        if (msg.sender != admin) revert NotAdmin();
        for (uint256 i = 0; i < banks.length; i++) {
            if (memberIndex[banks[i]] == 0) {
                members.push(banks[i]);
                memberIndex[banks[i]] = members.length;
            }
            agreementKey[banks[i]] = keys[i];
            emit MemberAdded(banks[i], keys[i]);
        }
    }

    function memberCount() external view returns (uint256) {
        return members.length;
    }

    modifier onlyMember() {
        if (memberIndex[msg.sender] == 0) revert NotMember();
        _;
    }

    // ------------------------------------------------------------------------------------------ phase 1: books
    function commitBooks(uint256 epoch, bytes32 root) external onlyMember {
        Epoch storage e = epochs[epoch];
        if (e.booksDone != 0) revert WrongPhase();
        if (bookRoot[epoch][msg.sender] != 0) revert Twice();
        if (e.booksOpened == 0) e.booksOpened = uint64(block.number);
        bookRoot[epoch][msg.sender] = root;
        emit BooksCommitted(epoch, msg.sender, root);
        if (++e.bookCount == members.length) e.booksDone = uint64(block.number);
    }

    /// @notice Close the books phase without the members that did not commit in time.
    function closeBooks(uint256 epoch) external {
        Epoch storage e = epochs[epoch];
        if (e.booksDone != 0 || e.booksOpened == 0 || block.number <= e.booksOpened + window) revert WrongPhase();
        e.booksDone = uint64(block.number);
    }

    // ------------------------------------------------------------------------------------------- phase 2: seed
    /// @notice The epoch seed: the hash of the first block after the books barrier, which no committed book
    ///         could depend on. Zero until that block exists.
    function seedOf(uint256 epoch) public view returns (bytes32) {
        Epoch storage e = epochs[epoch];
        if (e.seed != 0) return e.seed;
        if (e.booksDone == 0 || block.number <= e.booksDone + 1) return 0;
        return keccak256(abi.encodePacked(blockhash(e.booksDone + 1), epoch));
    }

    // ----------------------------------------------------------------------------------------- phase 3: sketch
    function commitSketch(uint256 epoch, bytes32 h) external onlyMember {
        Epoch storage e = epochs[epoch];
        if (bookRoot[epoch][msg.sender] == 0) revert NoCommitment();
        if (e.sketchesDone != 0) revert WrongPhase();
        if (e.seed == 0) {
            bytes32 s = seedOf(epoch);
            if (s == 0) revert WrongPhase();
            e.seed = s;     // fixed while blockhash is available (256 blocks)
            emit SeedDrawn(epoch, s);
        }
        if (sketchHash[epoch][msg.sender] != 0) revert Twice();
        sketchHash[epoch][msg.sender] = h;
        emit SketchCommitted(epoch, msg.sender, h);
        if (++e.sketchCount == e.bookCount) e.sketchesDone = uint64(block.number);
    }

    function closeSketches(uint256 epoch) external {
        Epoch storage e = epochs[epoch];
        if (e.sketchesDone != 0 || e.booksDone == 0 || block.number <= e.booksDone + 1 + window) revert WrongPhase();
        e.sketchesDone = uint64(block.number);
    }

    // ----------------------------------------------------------------------------------------- phase 4: reveal
    function _reveal(uint256 epoch, uint256[] calldata sketch) private {
        Epoch storage e = epochs[epoch];
        if (e.sketchesDone == 0 || block.number > e.sketchesDone + window) revert WrongPhase();
        if (sketch.length != 2 * cells) revert BadLength();
        bytes32 h = sketchHash[epoch][msg.sender];
        if (h == 0) revert NoCommitment();
        if (revealed[epoch][msg.sender]) revert Twice();
        if (keccak256(abi.encodePacked(sketch)) != h) revert Mismatch();
        revealed[epoch][msg.sender] = true;
        e.revealCount += 1;
    }

    function _add(uint256 epoch, uint256[] calldata sketch) private {
        mapping(uint256 => uint256) storage a = acc[epoch];
        uint256 n = 2 * cells;
        unchecked {
            for (uint256 i = 0; i < n; i += 2) {
                a[i] ^= sketch[i];
                uint256 o = a[i + 1];
                uint256 s = sketch[i + 1];
                uint256 cnt = ((o >> 128) + (s >> 128)) & LOW64;
                a[i + 1] = (cnt << 128) | ((o ^ s) & LOW128);
            }
        }
    }

    /// @notice Publish mode: the sketch travels in calldata and must match the commitment.
    function reveal(uint256 epoch, uint256[] calldata sketch) external onlyMember {
        _reveal(epoch, sketch);
        emit SketchRevealed(epoch, msg.sender, false);
    }

    /// @notice Accumulate mode: as reveal, and the sketch is added into the epoch's on-chain sum.
    function accumulate(uint256 epoch, uint256[] calldata sketch) external onlyMember {
        _reveal(epoch, sketch);
        _add(epoch, sketch);
        emit SketchRevealed(epoch, msg.sender, true);
    }

    /// @notice After the reveal window, a member that revealed posts the (signed) mask it shared with a member that
    ///         committed books but revealed no sketch; verifiers check the edge against the epoch's mask graph.
    function revealMask(uint256 epoch, address absent, uint256[] calldata mask, bool accumulated)
        external onlyMember
    {
        Epoch storage e = epochs[epoch];
        if (e.sketchesDone == 0 || block.number <= e.sketchesDone + window) revert WrongPhase();
        if (mask.length != 2 * cells) revert BadLength();
        if (!revealed[epoch][msg.sender] || revealed[epoch][absent] || bookRoot[epoch][absent] == 0) revert Mismatch();
        if (accumulated) _add(epoch, mask);
        emit MaskRevealed(epoch, absent, msg.sender);
    }

    // ------------------------------------------------------------------------------------------- commit mode
    /// @notice Post the sum of an epoch's committed sketches, once the sketch phase is closed.
    function postAggregate(uint256 epoch, uint256[] calldata sum) external onlyMember {
        if (sum.length != 2 * cells) revert BadLength();
        if (epochs[epoch].sketchesDone == 0) revert WrongPhase();
        if (aggregateHash[epoch] != 0) revert Twice();
        bytes32 h = keccak256(abi.encodePacked(sum));
        aggregateHash[epoch] = h;
        emit AggregatePosted(epoch, msg.sender, h);
    }

    // ------------------------------------------------------------------------------------------------- views
    /// @notice The epoch's on-chain sum (accumulate mode).
    function aggregate(uint256 epoch) external view returns (uint256[] memory out) {
        uint256 n = 2 * cells;
        out = new uint256[](n);
        mapping(uint256 => uint256) storage a = acc[epoch];
        for (uint256 i = 0; i < n; i++) out[i] = a[i];
    }

    /// @notice True when every member that committed books revealed, and the on-chain sum is empty.
    function clean(uint256 epoch) external view returns (bool complete, bool empty) {
        Epoch storage e = epochs[epoch];
        complete = e.sketchesDone != 0 && e.revealCount == e.bookCount && e.bookCount == members.length;
        empty = true;
        mapping(uint256 => uint256) storage a = acc[epoch];
        uint256 n = 2 * cells;
        for (uint256 i = 0; i < n; i++) {
            if (a[i] != 0) {
                empty = false;
                break;
            }
        }
    }
}
