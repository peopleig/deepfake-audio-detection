from pathlib import Path
import soundfile as sf
import concurrent.futures

def check_file(f):
    try:
        sf.read(str(f), dtype='float32', frames=1)
        return None
    except Exception:
        return str(f)

def parallel_clean(roots):
    files = []
    for root in roots:
        files.extend(list(root.rglob("*.wav")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_file, files))
    bad_files = [f for f in results if f is not None]
    return bad_files

if __name__ == "__main__":
    roots = [Path("data/for-norm/training/fake"), Path("data/for-rerecorded")]
    print("Starting parallel scanning...")
    bad_files = parallel_clean(roots)
    print(f"Found {len(bad_files)} corrupted files.")
    with open("corrupted_files.txt", "w") as f:
        for bf in bad_files:
            f.write(f"{bf}\n")
