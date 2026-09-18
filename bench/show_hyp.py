import json, sys, os
R = os.path.expanduser("~/asr_bench/results")
ids = sys.argv[1].split(",") if len(sys.argv) > 1 else ["vi_01"]
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sea-private", "sea-private-prod", "qwen17", "qwen17-prod"]
mani = {x["id"]: x for x in json.load(open(os.path.expanduser("~/asr_bench/manifest.json")))}
for i in ids:
    print(f"[REF          ] {i}: {mani[i]['ref'][:260]}")
    for tag in tags:
        p = f"{R}/{tag}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p))
        for it in d["items"]:
            if it["id"] == i:
                print(f"[{tag:13s}] CER {it['cer']:.1%} chunks={it['n_chunks']}")
                print("    ", it["hyp"][:260].replace("\n", " "))
    print()
