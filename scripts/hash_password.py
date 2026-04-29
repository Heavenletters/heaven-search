#!/usr/bin/env python3
"""Generate bcrypt password hash for the admin user.
Usage: python scripts/hash_password.py yourpassword
"""

import sys
from src.auth import hash_password

if len(sys.argv) < 2:
    print("Usage: python scripts/hash_password.py <password>")
    sys.exit(1)

print(hash_password(sys.argv[1]))
