# -*- coding: utf-8 -*-
"""
selftest_extra.py —— v3.1 新增自检项
由 run.py 的 mode_selftest 通过 register(cfg, check) 挂载：
  · 词汇册绑定→最近邻回路
  · 锻造试错的克隆环境隔离
  · 成长仪表盘生成
"""
from __future__ import annotations

import os


def register(cfg, check) -> None:
    # ---- 词汇册：绑定 → 最近邻 ----
    def t_lexicon():
        from evolution_engine import Lexicon
        lx = Lexicon()
        pat = [1.0, 0.0, -1.0, 0.5, 0.0, 0.2, -0.3, 0.0]
        lx.bind("危险", pat, 10)
        hit = lx.nearest(pat)
        assert hit and hit["word"] == "危险" and hit["sim"] > 0.99, hit
        low = lx.nearest([-x for x in pat])       # 反相模式应无近亲
        assert low is None or low["sim"] < 0.9, low
        got = Lexicon.extract_word("教：「风暴」要来了")
        assert got == "风暴", got
        lx.bind("危险", pat, 20)                  # 重教 = 巩固，不重复建条目
        assert len(lx) == 1 and lx.entries[0]["uses"] == 2
    check("词汇册绑定与最近邻", t_lexicon)

    # ---- 锻造试错：克隆环境带独立随机源且稳定势有界 ----
    def t_forge_env():
        import random as _r
        from evolution_engine import ResourceEnv
        eng = SwarmEngine(EngineConfig(
            seed=21, n0=32,
            outdir=os.path.join(cfg.outdir, "_selftest")))
        env2 = eng.env.clone(_r.Random(eng.rng.randrange(2 ** 31)))
        for _ in range(5):
            env2.step()
        assert 0.0 <= env2.potential() <= 1.0
    check("锻造试错环境隔离", t_forge_env)

    # ---- 成长仪表盘生成 ----
    def t_dashboard():
        base = os.path.join(cfg.outdir, "_selftest")
        path = build_dashboard(base)
        assert os.path.exists(path), path
        with open(path, encoding="utf-8") as fh:
            txt = fh.read()
        assert "<svg" in txt and "成长仪表盘" in txt
    check("成长仪表盘生成", t_dashboard)


from dashboard import build_dashboard          # noqa: E402
from evolution_engine import EngineConfig, SwarmEngine  # noqa: E402
