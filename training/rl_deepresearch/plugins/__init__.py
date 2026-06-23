"""slime plugin package: the ONLY code we write for RL training.

slime (the engine) is consumed as a Docker image and never forked. It calls into
this package by CLI flag:

    --custom-generate-function-path plugins.rollout.generate
    --custom-rm-path                plugins.reward.reward_func

Both functions reuse libs/dr_agent so the RL environment and reward are identical
to what eval/ and apps/ use — no train/eval skew.
"""
