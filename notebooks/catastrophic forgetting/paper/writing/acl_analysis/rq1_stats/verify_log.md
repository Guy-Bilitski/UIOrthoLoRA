# rq1_stats verification log

verdict: ALL OK

OK   V1 row count matches -- 26 committed vs 26 recomputed
OK   V1 d_ret agrees (|diff|<=5e-3) -- max diff 3.33e-05, n=26
OK   V1 d_ret NaN pattern matches
OK   V1 p_ret agrees (|diff|<=5e-3) -- max diff 4.87e-05, n=25
OK   V1 p_ret NaN pattern matches
OK   V1 d_adapt agrees (|diff|<=5e-3) -- max diff 3.33e-05, n=26
OK   V1 d_adapt NaN pattern matches
OK   V1 p_adapt agrees (|diff|<=5e-3) -- max diff 5.00e-05, n=25
OK   V1 p_adapt NaN pattern matches
OK   V1 Holm(all,ret) agrees -- max diff 4.82e-05
OK   V1 Holm(all,ret) NaN pattern matches
OK   V1 Holm(all,adapt) agrees -- max diff 4.84e-05
OK   V1 Holm(all,adapt) NaN pattern matches
OK   V1 verdict counts consistent -- better 0=0, worse 2=2
OK   V1 SC-LoRA qwen-math n.s. after Holm -- raw 0.0352, holm-all 0.4582
OK   V2 pooled TOST model reproduces (beta, se, flags) -- 9 methods, G=343
OK   V2 equiv_1pp flags match stored 90% CIs
OK   V2 equiv_2pp flags match stored 90% CIs
OK   V2 equiv_3pp flags match stored 90% CIs
OK   V3 MDE analytic == stored, simulated power ~0.8 -- llama_cs/SC-LoRA: power 0.801; qwen_cs/SC-LoRA: power 0.800; qwen_math/MiLoRA: power 0.796
