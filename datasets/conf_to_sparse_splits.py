import argparse
import json
import os


def write_sparse_splits(conf_json_path, outdir):
    """Derive 3dgs-go's sparse_<N>.txt / sparse_test.txt (0-indexed positions into
    the alphabetically-sorted COLMAP camera list) from a dataset_splits conf.json
    (1-indexed CamNNN camera numbers, as used by the zipnerf ActorsHQ loader).

    Both loaders reference the same underlying camera, sorted the same way (by
    filename) -- the only difference is the indexing convention, a constant -1
    offset (verified against the released mip360/garden and ActorsHQ splits)."""
    with open(conf_json_path) as f:
        conf = json.load(f)

    train_cameras = [int(x) for x in conf["train_cameras"].split()]
    val_cameras = [int(x) for x in conf["val_cameras"].split()]

    os.makedirs(outdir, exist_ok=True)

    sparse_n_path = os.path.join(outdir, f"sparse_{len(train_cameras)}.txt")
    with open(sparse_n_path, "w") as f:
        f.write("\n".join(str(c - 1) for c in train_cameras) + "\n")

    sparse_test_path = os.path.join(outdir, "sparse_test.txt")
    with open(sparse_test_path, "w") as f:
        f.write("\n".join(str(c - 1) for c in val_cameras) + "\n")

    print(f"Wrote {sparse_n_path} ({len(train_cameras)} cameras)")
    print(f"Wrote {sparse_test_path} ({len(val_cameras)} cameras)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Derive 3dgs-go's sparse_<N>.txt / sparse_test.txt view-index files "
                    "from a dataset_splits/<dataset>/<scene>/<camcount>/conf.json, so the "
                    "camera split only needs to be defined once.")
    parser.add_argument("conf_json", help="Path to a dataset_splits .../<camcount>/conf.json")
    parser.add_argument("outdir", help="COLMAP dataset directory to write sparse_<N>.txt / sparse_test.txt into "
                                       "(e.g. datasets/3dgs-go-dataset/Actor01)")
    opt = parser.parse_args()

    write_sparse_splits(opt.conf_json, opt.outdir)
