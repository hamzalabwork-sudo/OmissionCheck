// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

/// @title OmissionOracle
/// @notice Consortium-wide omission check for interbank audit evidence. Every member bank publishes, per epoch, a
///         masked linear sketch (an invertible Bloom lookup table) of the interbank transfers in its books: +1 for
///         transfers it paid, -1 for transfers it received. Pairwise masks cancel in the sum, and so does every
///         transfer that both witnesses recorded identically, so the sum of all sketches holds only the
///         discrepancies (omitted, fabricated or altered transfers). Anyone can read the sum and decode it.
///
///         Cell layout: two 256-bit words per cell. Word 0 is the XOR of the 256-bit keys. Word 1 packs a signed
///         count in bits 64..127 (added modulo 2^64) and a 64-bit check value in bits 0..63 (XOR).
///
///         Three publication modes:
///           accumulate: the contract adds each sketch into on-chain storage (small consortia, fee-free chains);
///           publish:    the sketch travels in calldata, only its hash is stored, and the sum is formed off chain
///                       from the transaction data;
///           commit:     only the hash is put on chain and the sketch is exchanged off chain; one member posts
///                       the sum, which anyone holding the sketches can check against the committed hashes
///                       (large consortia: on-chain state is 32 bytes per bank plus one sketch per epoch).
contract OmissionOracle {
    address public immutable admin;
    uint256 public immutable cells;          // m, cells per sketch

    address[] public members;
    mapping(address => uint256) public memberIndex;   // 1-based
    mapping(address => bytes32) public agreementKey;  // X25519 public key used to derive pairwise keys

    mapping(uint256 => mapping(uint256 => uint256)) private acc;          // epoch => word => value
    mapping(uint256 => mapping(address => bytes32)) public sketchHash;    // epoch => member => keccak(sketch)
    mapping(uint256 => uint256) public submissions;                       // epoch => members that published

    uint256 private constant LOW64 = 0xFFFFFFFFFFFFFFFF;

    event MemberAdded(address indexed bank, bytes32 agreementKey);
    event SketchPublished(uint256 indexed epoch, address indexed bank, bytes32 sketchHash, bool accumulated);

    error NotAdmin();
    error NotMember();
    error AlreadyPublished();
    error BadLength();

    constructor(uint256 m) {
        admin = msg.sender;
        cells = m;
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

    function _register(uint256 epoch, bytes32 h, bool accumulated) private {
        if (memberIndex[msg.sender] == 0) revert NotMember();
        if (sketchHash[epoch][msg.sender] != 0) revert AlreadyPublished();
        sketchHash[epoch][msg.sender] = h;
        submissions[epoch] += 1;
        emit SketchPublished(epoch, msg.sender, h, accumulated);
    }

    /// @notice Add a sketch (2m words) into the epoch's on-chain sum.
    function accumulate(uint256 epoch, uint256[] calldata sketch) external {
        uint256 n = 2 * cells;
        if (sketch.length != n) revert BadLength();
        _register(epoch, keccak256(abi.encodePacked(sketch)), true);
        mapping(uint256 => uint256) storage a = acc[epoch];
        unchecked {
            for (uint256 i = 0; i < n; i += 2) {
                a[i] ^= sketch[i];
                uint256 o = a[i + 1];
                uint256 s = sketch[i + 1];
                uint256 cnt = ((o >> 64) + (s >> 64)) & LOW64;
                a[i + 1] = (cnt << 64) | ((o ^ s) & LOW64);
            }
        }
    }

    /// @notice Publish a sketch in calldata; only its hash is stored.
    function publish(uint256 epoch, uint256[] calldata sketch) external {
        if (sketch.length != 2 * cells) revert BadLength();
        _register(epoch, keccak256(abi.encodePacked(sketch)), false);
    }

    /// @notice Commit to a sketch exchanged off chain.
    function commit(uint256 epoch, bytes32 h) external {
        _register(epoch, h, false);
    }

    mapping(uint256 => bytes32) public aggregateHash;   // commit mode: hash of the posted sum
    event AggregatePosted(uint256 indexed epoch, address indexed poster, bytes32 aggregateHash);

    /// @notice Post the sum of an epoch's committed sketches (commit mode), once every member has committed.
    function postAggregate(uint256 epoch, uint256[] calldata sum) external {
        if (memberIndex[msg.sender] == 0) revert NotMember();
        if (sum.length != 2 * cells) revert BadLength();
        if (submissions[epoch] != members.length || aggregateHash[epoch] != 0) revert AlreadyPublished();
        bytes32 h = keccak256(abi.encodePacked(sum));
        aggregateHash[epoch] = h;
        emit AggregatePosted(epoch, msg.sender, h);
    }

    /// @notice The epoch's on-chain sum (accumulate mode).
    function aggregate(uint256 epoch) external view returns (uint256[] memory out) {
        uint256 n = 2 * cells;
        out = new uint256[](n);
        mapping(uint256 => uint256) storage a = acc[epoch];
        for (uint256 i = 0; i < n; i++) out[i] = a[i];
    }

    /// @notice True when every member published and the sum is empty: no discrepancy in the epoch.
    function clean(uint256 epoch) external view returns (bool complete, bool empty) {
        complete = submissions[epoch] == members.length;
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
