pragma circom 2.1.6;

// Sketch binding: the published (masked) cells of a bank are the masked field sketch of the books whose root the bank
// committed before the epoch seed. Mirrors omission/zk/circuit.py (the R1CS model used for the counts in the paper).
// NOT YET COMPILED: circom and snarkjs could not be installed in the session that wrote it. The Python reference uses
// prototype Poseidon constants; switch omission/zk/field.py to circomlib's constants before comparing witnesses.

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/comparators.circom";

// 4-ary Poseidon Merkle root of n leaves (n a power of 4)
template MerkleRoot4(n) {
    signal input leaves[n];
    signal output root;
    var nodes = (n - 1) \ 3;
    component h[nodes];
    signal lv[n + nodes];
    for (var i = 0; i < n; i++) { lv[i] <== leaves[i]; }
    var base = 0; var next = n; var cnt = n; var k = 0;
    while (cnt > 1) {
        for (var j = 0; j < cnt \ 4; j++) {
            h[k] = Poseidon(4);
            for (var t = 0; t < 4; t++) { h[k].inputs[t] <== lv[base + 4 * j + t]; }
            lv[next + j] <== h[k].out;
            k++;
        }
        base = next; next = next + cnt \ 4; cnt = cnt \ 4;
    }
    root <== lv[base];
}

// commitment to a fixed-length vector: a chain of width-17 permutations absorbing 15 elements each
template Commit(len, domain) {
    signal input in[len];
    signal output out;
    var C = (len + 14) \ 15;
    component p[C];
    for (var k = 0; k < C; k++) {
        p[k] = Poseidon(16);
        if (k == 0) { p[k].inputs[0] <== domain; } else { p[k].inputs[0] <== p[k - 1].out; }
        for (var i = 0; i < 15; i++) {
            if (15 * k + i < len) { p[k].inputs[i + 1] <== in[15 * k + i]; } else { p[k].inputs[i + 1] <== 0; }
        }
    }
    out <== p[C - 1].out;
}

// keyed sponge stream: 16 rate elements per permutation, the capacity element is never output
template Stream(len) {
    signal input key;
    signal input nonce;
    signal output out[len];
    var S = (len + 15) \ 16;
    component p[S];
    for (var k = 0; k < S; k++) {
        p[k] = PoseidonEx(16, 17);
        if (k == 0) {
            p[k].initialState <== 0;
            p[k].inputs[0] <== key;
            p[k].inputs[1] <== nonce;
            for (var i = 2; i < 16; i++) { p[k].inputs[i] <== 0; }
        } else {
            p[k].initialState <== p[k - 1].out[0];
            for (var i = 0; i < 16; i++) { p[k].inputs[i] <== p[k - 1].out[i + 1]; }
        }
        for (var i = 0; i < 16; i++) {
            if (16 * k + i < len) { out[16 * k + i] <== p[k].out[i + 1]; }
        }
    }
}

// n padded book entries, m = 192 cells (3 partitions of 64), kappa mask slots (kappa/2 with +1, then kappa/2 with -1)
template SketchBinding(n, kappa) {
    var m = 192; var B = 6; var K = 3;
    var DOM_CELLS = 1397441876;   // 0x534B4554
    var DOM_PAIR = 1346455890;    // 0x50414952

    signal input bank;
    signal input root;
    signal input seed;
    signal input cellsCommitment;
    signal input nonce;
    signal input pairCommitment[kappa];

    signal input key[n];
    signal input w[n];
    signal input salt[n];
    signal input cells[3 * m];
    signal input pairKey[kappa];

    // 1. salted leaves hash to the committed root; the weight follows from the key's payer and payee banks
    component lf[n];
    component mr = MerkleRoot4(n);
    component kb[n]; component ip[n]; component iq[n];
    for (var i = 0; i < n; i++) {
        lf[i] = Poseidon(3);
        lf[i].inputs[0] <== salt[i];
        lf[i].inputs[1] <== key[i];
        lf[i].inputs[2] <== w[i];
        mr.leaves[i] <== lf[i].out;
        kb[i] = Num2Bits(248);            // key = payer (32) || payee (32) || tag (184)
        kb[i].in <== key[i];
        var payer = 0; var payee = 0;
        for (var b = 0; b < 32; b++) {
            payer += kb[i].out[216 + b] * (1 << b);
            payee += kb[i].out[184 + b] * (1 << b);
        }
        ip[i] = IsZero(); ip[i].in <== payer - bank;
        iq[i] = IsZero(); iq[i].in <== payee - bank;
        w[i] === ip[i].out - iq[i].out;
    }
    mr.root === root;

    // 3a. the witness cells are the committed ones; challenges drawn from the commitments
    component cc = Commit(3 * m, DOM_CELLS);
    for (var c = 0; c < 3 * m; c++) { cc.in[c] <== cells[c]; }
    cc.out === cellsCommitment;
    component cr = Poseidon(4);
    component ca = Poseidon(4);
    cr.inputs[0] <== cellsCommitment; cr.inputs[1] <== root; cr.inputs[2] <== seed; cr.inputs[3] <== 1;
    ca.inputs[0] <== cellsCommitment; ca.inputs[1] <== root; ca.inputs[2] <== seed; ca.inputs[3] <== 2;
    signal r2[8];
    r2[0] <== cr.out;
    for (var k = 1; k < 8; k++) { r2[k] <== r2[k - 1] * r2[k - 1]; }
    signal al2;
    al2 <== ca.out * ca.out;

    // 2. per entry: keyed hash, strict bits (unique positions), r^position, contribution
    component eh[n];
    component nb[n];
    signal t[n][K][B];
    signal acc[n][K][B + 1];
    signal u[n]; signal v[n]; signal q[n]; signal contrib[n];
    var rhs = 0;
    for (var i = 0; i < n; i++) {
        eh[i] = Poseidon(2);
        eh[i].inputs[0] <== seed;
        eh[i].inputs[1] <== key[i];
        nb[i] = Num2Bits_strict();
        nb[i].in <== eh[i].out;
        for (var j = 0; j < K; j++) {
            if (j == 0) { acc[i][j][0] <== 1; } else if (j == 1) { acc[i][j][0] <== r2[6]; } else { acc[i][j][0] <== r2[7]; }
            for (var b = 0; b < B; b++) {
                t[i][j][b] <== nb[i].out[B * j + b] * (r2[b] - 1);
                acc[i][j][b + 1] <== acc[i][j][b] * (1 + t[i][j][b]);
            }
        }
        u[i] <== ca.out * key[i];
        v[i] <== al2 * eh[i].out;
        q[i] <== (1 + u[i] + v[i]) * (acc[i][0][B] + acc[i][1][B] + acc[i][2][B]);
        contrib[i] <== w[i] * q[i];
        rhs += contrib[i];
    }

    // 3b. masks from the committed pair keys
    component pc[kappa];
    component st[kappa];
    for (var b = 0; b < kappa; b++) {
        pc[b] = Poseidon(2);
        pc[b].inputs[0] <== pairKey[b];
        pc[b].inputs[1] <== DOM_PAIR;
        pc[b].out === pairCommitment[b];
        st[b] = Stream(3 * m);
        st[b].key <== pairKey[b];
        st[b].nonce <== nonce;
    }

    // 4. random evaluation of the unmasked cells equals that of the sketch of the committed books
    signal rc[m];
    signal x1[m]; signal x2[m]; signal term[m];
    rc[0] <== 1;
    rc[1] <== r2[0];
    for (var c = 2; c < m; c++) { rc[c] <== rc[c - 1] * r2[0]; }
    var lhs = 0;
    for (var c = 0; c < m; c++) {
        var du = cells[c];
        var dv = cells[m + c];
        var dz = cells[2 * m + c];
        for (var b = 0; b < kappa; b++) {
            var sg = b < kappa \ 2 ? 1 : -1;
            du -= sg * st[b].out[c];
            dv -= sg * st[b].out[m + c];
            dz -= sg * st[b].out[2 * m + c];
        }
        x1[c] <== ca.out * dv;
        x2[c] <== al2 * dz;
        term[c] <== rc[c] * (du + x1[c] + x2[c]);
        lhs += term[c];
    }
    lhs === rhs;
}

component main {public [bank, root, seed, cellsCommitment, nonce, pairCommitment]} = SketchBinding(64, 16);
