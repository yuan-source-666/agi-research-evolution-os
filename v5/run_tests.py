"""零依赖测试运行器：python run_tests.py（亦兼容 pytest）。"""
import sys
import traceback
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

passed = failed = 0
failures = []

for tf in sorted((root / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(tf.stem, tf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        try:
            fn()
            passed += 1
            print(f"PASS  {tf.stem}::{name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            failures.append((f"{tf.stem}::{name}", traceback.format_exc()))
            print(f"FAIL  {tf.stem}::{name}: {e}")

print(f"\n===== {passed} passed, {failed} failed =====")
for name, tb in failures:
    print("\n" + "-" * 60 + f"\n{name}\n{tb}")
sys.exit(1 if failed else 0)
