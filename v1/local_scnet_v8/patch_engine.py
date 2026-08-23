# -*- coding: utf-8 -*-
"""evolution_loop.py 引擎补丁（首次点火发现的两处缺陷修复）"""
import shutil, sys

SRC = '/root/private_data/evolution_loop.py'
BAK = '/root/private_data/evolution_loop.py.bak_20260822'

shutil.copy2(SRC, BAK)
print('backup ->', BAK)

src = open(SRC, encoding='utf-8').read()

# ---- Fix 1: rollback 只回滚 target 版本之后引入的改进（沿版本链收集），且按接受逆序执行 ----
old1 = '''            # Execute rollback hooks
            for imp in self.improvements:
                if imp.status == "accepted" and imp.id in self._rollback_hooks:
                    try:
                        self._rollback_hooks[imp.id]()
                        self._log_evolution("rollback", f"Executed rollback hook: {imp.id}")
                    except Exception as e:
                        self._log_evolution("rollback", f"Rollback hook failed {imp.id}: {e}")'''
new1 = '''            # Execute rollback hooks: only undo improvements introduced
            # AFTER the target version (walk the version chain from current
            # back to target), in reverse order of acceptance.
            undo_ids = set()
            v = self.current_version
            while v and v.version_id != target.version_id:
                undo_ids.update(v.improvements)
                v = next((x for x in self.versions if x.version_id == v.parent_version), None)
            self._log_evolution("rollback", f"Undo set: {sorted(undo_ids) if undo_ids else 'empty'}")
            for imp in reversed(self.improvements):
                if imp.id in undo_ids and imp.id in self._rollback_hooks:
                    try:
                        self._rollback_hooks[imp.id]()
                        self._log_evolution("rollback", f"Executed rollback hook: {imp.id}")
                    except Exception as e:
                        self._log_evolution("rollback", f"Rollback hook failed {imp.id}: {e}")'''
assert src.count(old1) == 1, f'Fix1 anchor not unique: {src.count(old1)}'
src = src.replace(old1, new1)

# ---- Fix 2: 无验证证据的改进不得默认放行（skipped != passed）----
old2 = '''            else:
                # Default independent eval: check sandbox + benchmark
                sandbox_ok = improvement.sandbox_result.get("status") != "error" and \\
                             "error" not in improvement.sandbox_result
                bench_ok = "error" not in improvement.benchmark_result
                improvement.eval_result = {
                    "sandbox_pass": sandbox_ok,
                    "benchmark_pass": bench_ok,
                    "overall": sandbox_ok and bench_ok
                }'''
new2 = '''            else:
                # Default independent eval: check sandbox + benchmark.
                # NOTE (fix 2026-08-22): "skipped" is NOT a pass. An
                # improvement without real validation evidence is rejected.
                sandbox_ok = (improvement.sandbox_result.get("status") == "ok"
                              and "error" not in improvement.sandbox_result)
                bench_ok = (bool(improvement.benchmark_result)
                            and "error" not in improvement.benchmark_result
                            and improvement.benchmark_result.get("status") != "skipped")
                improvement.eval_result = {
                    "sandbox_pass": sandbox_ok,
                    "benchmark_pass": bench_ok,
                    "overall": sandbox_ok and bench_ok
                }'''
assert src.count(old2) == 1, f'Fix2 anchor not unique: {src.count(old2)}'
src = src.replace(old2, new2)

open(SRC, 'w', encoding='utf-8').write(src)
print('patched both fixes OK')

# 语法自检
import py_compile
py_compile.compile(SRC, doraise=True)
print('syntax OK')
