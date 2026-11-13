"""/workspace cannot hold symlinks, so HF snapshot files there are empty. Rebuild a snapshot on local disk with symlinks into the blobs.
  python link_model.py google/gemma-4-31B-it /root/models/gemma-4-31B-it"""
import os, sys
from huggingface_hub import HfApi

repo, dest = sys.argv[1], sys.argv[2]
blobs = f"/workspace/.cache/huggingface/hub/models--{repo.replace('/', '--')}/blobs"
os.makedirs(dest, exist_ok=True)
for f in HfApi().list_repo_tree(repo, recursive=True, expand=True):
    if not hasattr(f, "blob_id"):
        continue
    blob = f.lfs.sha256 if f.lfs else f.blob_id
    src = os.path.join(blobs, blob)
    if not os.path.exists(src):
        print("missing", f.path)
        continue
    assert os.path.getsize(src) == f.size, f.path
    out = os.path.join(dest, f.path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.lexists(out):
        os.remove(out)
    os.symlink(src, out)
    print("linked", f.path, f.size)
