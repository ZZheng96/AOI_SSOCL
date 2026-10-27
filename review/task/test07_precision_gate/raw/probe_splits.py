import sqlite3, os, json
db = r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\database\aoi.db"
c = sqlite3.connect("file:" + db + "?mode=ro", uri=True)
rows = c.execute("""SELECT split,label,width,height,COUNT(*) FROM images
                    WHERE category='gold_finger' GROUP BY split,label,width,height""").fetchall()
for r in rows:
    print(r)
print("---existence check (test/train/train_anomaly)---")
for split in ("test", "train", "train_anomaly"):
    rs = c.execute("SELECT id,path,label,width,height FROM images WHERE category='gold_finger' AND split=?",
                   (split,)).fetchall()
    miss = [r for r in rs if not os.path.exists(r[1])]
    print(split, "n=", len(rs), "missing=", len(miss),
          "labels=", {r[2] for r in rs}, "sizes=", {(r[3], r[4]) for r in rs})
c.close()
