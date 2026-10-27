import sqlite3, os, glob, sys

root = sys.argv[1] if len(sys.argv) > 1 else r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core"
seen = set()
for p in glob.glob(os.path.join(root, "**", "*.db"), recursive=True):
    rp = os.path.realpath(p)
    if rp in seen:
        continue
    seen.add(rp)
    try:
        c = sqlite3.connect("file:" + rp + "?mode=ro", uri=True)
        tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
        print("==", rp.replace(root, ""), tables[:14])
        if "models" in tables:
            for r in c.execute("select id,category,version,is_active from models limit 30"):
                print("   model", r)
        if "images" in tables:
            for r in c.execute("select category,count(*) from images group by category"):
                print("   images", r)
        if "detections" in tables:
            print("   detections", c.execute("select count(*) from detections").fetchone())
        c.close()
    except Exception as e:
        print(rp, "ERR", e)
