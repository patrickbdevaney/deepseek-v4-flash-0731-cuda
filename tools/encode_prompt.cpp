// encode_prompt.cpp — turn text into the token-id list `build/decode` takes as argv[2].
//
// Why this exists: every full-model measurement so far has used ONE prompt
// (`0,671,6102,294,8760,344`), and Finding 49's adaptive-verify threshold was fitted on 18 verifies
// of it. Validating that default on other prompts needs their ids, and INVENTING ids is exactly the
// class of mistake this project has a rule against. So the ids come from the checkpoint's own
// tokenizer, and the tool GATES itself on reproducing the canonical prompt before printing anything
// else: if `The capital of France is` does not come back as 671,6102,294,8760,344, the encoder does
// not match this checkpoint and its output must not be used.
//
//   build: g++ -O2 -std=c++17 -I include tools/encode_prompt.cpp -o build/encode_prompt
//   run:   ./build/encode_prompt <ckpt>/tokenizer.json "text one" "text two"
#include "tokenizer.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <tokenizer.json> [text...]\n", argv[0]); return 2; }
    Tokenizer t;
    t.load(argv[1]);
    printf("[encode] vocab %zu  merges %zu  specials %zu\n", t.id2tok.size(), t.merges.size(), t.specials.size());

    // --- GATE: the canonical prompt, whose ids are recorded in run_model.sh and DECODE_FLYWHEEL.md ---
    const std::vector<int> want = {671, 6102, 294, 8760, 344};
    std::vector<int> got = t.encode("The capital of France is", false);
    bool pass = (got == want);
    printf("[encode] GATE canonical: got");
    for (int v : got) printf(" %d", v);
    printf("   want");
    for (int v : want) printf(" %d", v);
    printf("  -> %s\n", pass ? "GATE PASS" : "GATE FAIL");
    if (!pass) {
        fprintf(stderr, "[encode] encoder does not reproduce the canonical prompt on this checkpoint;\n"
                        "[encode] its ids are NOT usable. Refusing to print prompts.\n");
        return 1;
    }

    // BOS is id 0 here: the canonical argv is `0,` + the encoded text.
    for (int i = 2; i < argc; ++i) {
        std::vector<int> ids = t.encode(argv[i], false);
        printf("0");
        for (int v : ids) printf(",%d", v);
        printf("      # (%zu ids) %s\n", ids.size() + 1, argv[i]);
        printf("[encode]   round-trip: \"%s\"\n", t.decode(ids, true).c_str());
    }
    return 0;
}
