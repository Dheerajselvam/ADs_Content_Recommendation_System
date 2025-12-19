import hashlib

def stable_hash(x, salt="rec"):
    h = hashlib.md5(f"{x}-{salt}".encode()).hexdigest()
    return int(h[:8], 16)