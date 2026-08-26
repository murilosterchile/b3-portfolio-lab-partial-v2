from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/services/api")

from app.db import Base, SessionLocal, engine  # noqa: E402
from app.demo import seed_demo  # noqa: E402

Base.metadata.create_all(engine)
with SessionLocal() as db:
    count = seed_demo(db)
print(f"seeded={count}")
