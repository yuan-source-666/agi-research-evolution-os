# -*- coding: utf-8 -*-
"""补丁 #3：全量回归门禁内化为引擎默认准则。

背景：首火实验中，坏改进 ImpD 被「子集阈值过松的缺陷基准」放行，
靠外部全量回归检查兜底才回滚。本补丁把这道防线内化进 decide()：
任何 VERIFIED 改进在接受前必须通过已注册的 regression gate
（全量任务集 + 全新种子），否则当场拒绝，不进入版本链。
"""
import shutil
import py_compile

PATH = '/root/private_data/evolution_loop.py'
BAK = PATH + '.bak2_20260822'

src = open(PATH, encoding='utf-8').read()

assert 'set_regression_gate' not in src, 'patch #3 already applied'

# ---- 1) decide() 前置全量回归门禁 ----
A1 = '''            accepted = False

            if improvement.evidence == EvidenceLevel.VERIFIED.value:
                # Validation passed -> accept
                improvement.status = "accepted"'''
A1_NEW = '''            accepted = False

            # (fix 2026-08-22 #3) Full-regression gate: benchmark evidence may
            # come from a subset or a flawed threshold. Before accepting any
            # VERIFIED improvement, re-evaluate on the FULL suite with a FRESH
            # seed via the registered gate. Blocked -> rejected on the spot,
            # never enters the version chain.
            gate = getattr(self, 'regression_gate', None)
            if improvement.evidence == EvidenceLevel.VERIFIED.value and gate is not None:
                try:
                    gate_result = gate(improvement) or {}
                except Exception as e:
                    gate_result = {'overall': False, 'detail': 'gate error: %s' % e}
                improvement.eval_result = dict(improvement.eval_result or {})
                improvement.eval_result['regression_gate'] = gate_result
                if not gate_result.get('overall'):
                    improvement.evidence = EvidenceLevel.FAILED.value
                    improvement.status = 'rejected'
                    self._log_evolution('decide',
                        'Regression gate BLOCKED: %s -- %s' % (improvement.title,
                                                               gate_result.get('detail')))
                    if self.memory:
                        self.memory.store_episodic(
                            'Improvement blocked by regression gate: %s -- %s'
                            % (improvement.title, gate_result.get('detail')),
                            source='evolution_loop', importance=0.7,
                            tags=['evolution', 'regression_gate'], evidence='FAILED')
                    improvement.decided_at = time.time()
                    return False

            if improvement.evidence == EvidenceLevel.VERIFIED.value:
                # Validation passed -> accept
                improvement.status = "accepted"'''
assert src.count(A1) == 1, 'anchor 1 not unique: %d' % src.count(A1)
src = src.replace(A1, A1_NEW)

# ---- 2) set_regression_gate API ----
A2 = '    # ========== Phase 6: Rollback =========='
A2_NEW = '''    def set_regression_gate(self, gate_fn):
        """(fix 2026-08-22 #3) 注册全量回归门禁，decide() 接受前强制调用。"""
        self.regression_gate = gate_fn

    # ========== Phase 6: Rollback =========='''
assert src.count(A2) == 1, 'anchor 2 not unique'
src = src.replace(A2, A2_NEW)

shutil.copyfile(PATH, BAK)
open(PATH, 'w', encoding='utf-8').write(src)
py_compile.compile(PATH, doraise=True)
print('patch #3 applied OK. backup: %s' % BAK)
